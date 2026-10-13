# BlindAssist

BlindAssist is an Android showcase research prototype for goal-driven visual
assistance. It combines camera perception, risk logic, and concise guidance to
demonstrate measurable effects in clearly stated controlled conditions. It is
not a certified mobility or safety product.

App v10.19.2 opens the original home without starting perception. **Start assist**
connects the camera and ToF. Current H3 CNH firmware uses **basic range assistance**;
the legacy mapped-time transport can use **A+LOCAL experimental mode**, with
the frozen research classifier running offline on the phone. Its nominal camera/ToF
registration is not physical calibration; simulation results do not establish hardware
accuracy. The original ToF mode remains available. See [hardware guide](docs/HARDWARE_OBSTACLE_DEMO.md).

The obstacle-perception research goal is **盲杖互补的类别无关前视障碍感知**
(cane-complementary, class-agnostic forward obstacle awareness). Priorities are
forward walls and large obstructions, body/head protrusions, suspended hazards,
and multi-height poles, with useful direction and coarse range evidence under
limited compute. The wearer chooses movement; this is not autonomous avoidance.
Very low obstacles remain secondary compatibility evidence, not a headline
optimization target. Knee-height hazards are not discarded. Earlier warning and
dynamic-obstacle coverage are objectives requiring temporal validation.

The current research design combines base near-obstacle evidence with optional
ground-relative enhancement. Missing ground may remove height refinement but
must not erase an already observed near obstacle. This design and the new task
priorities are research direction, not a claim that all capabilities are already
deployed or validated. See the [current perception route](research/active/dtr-r0/CURRENT.md).

## Start here

- **当前手机版本 60 / 10.19.2**：实时预览与开始／停止相邻，导出任务和连接记录下移；两板固件优先连接已配置手机热点，并保留原 Wi-Fi 回退。[热点与采集页更新](research/active/hardware-bringup/HOTSPOT_PRIORITY_20261013.md)
- **融合实验待佩戴验证**：“数据采集 → 打开融合实验 → 开始实验”，显示 ToF 强档或 RGB 的 HEAD/BODY 融合，经 2/3 帧确认；无语音震动，不替换默认基础 ToF。已补因果输入及两项数值修复，真实流 RGB 分支与完整融合仍待有效姿态下验证。[实验入口与状态](docs/HARDWARE_OBSTACLE_DEMO.md)
- **手机端 Atom RGB + ToF + IMU 采集**：首页“数据采集”同时预览 Atom 画面、ToF 距离网格和 IMU 读数，保存原始 JPEG、完整 8×8×16 CNH、Atom 六轴与姿态、时间戳和场景备注；停止后可导出 ZIP。使用局域网或已配置的手机热点。[采集使用说明](docs/PHONE_DATA_CAPTURE.md)
- 默认入口（v10.19.2）：打开 App 停留首页；点击“开始辅助”连接相机与 ToF，当前 H3 固件使用基础测距，旧版时序协议保留 A+LOCAL 实验模式。[使用说明](docs/HARDWARE_OBSTACLE_DEMO.md)。首页语言、提醒频率、语音风格、震动强度和关怀模式作用于辅助页；结束后停止连接与反馈。支持最近 20 秒留证、静音回放与分路降级。
- **中文一页状态（系统、指标、当前问题）**: [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md)
- Current research decision: [docs/CURRENT_DECISION.md](docs/CURRENT_DECISION.md)
- L10-R0 paused / retained evidence: [research/active/l10-r0/CURRENT.md](research/active/l10-r0/CURRENT.md)
- Forward obstacle perception / retained DTR history: [research/active/dtr-r0/CURRENT.md](research/active/dtr-r0/CURRENT.md)
- Code ownership: [docs/CODE_MAP.md](docs/CODE_MAP.md)
- Documentation map: [docs/README.md](docs/README.md)
- Historical lookup: [docs/history-index.md](docs/history-index.md)

Forward obstacle perception is the sole advancing research line. L10 is paused;
its results and reproduction paths are retained. Closed
experiments are summarized in `experiments/index.jsonl` and remain recoverable
from the remote tag `archive/pre-agent-surface-2026-08-26` or the terminal
commits listed in `docs/history-index.md`.

## Workstation profiles

Copy the local template once and edit only machine-owned values:

```powershell
Copy-Item config/local.example.toml config/local.toml
pwsh -NoProfile -File tools/ba.ps1 setup base
pwsh -NoProfile -File tools/ba.ps1 doctor base
```

Profiles are independent: `base`, `research-l10-r0`, `research-dtr-r0`,
`android`, `device`, and `export`. Research setup does not install or probe
Android tooling.

```powershell
pwsh -NoProfile -File tools/ba.ps1 setup research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 doctor research-dtr-r0
pwsh -NoProfile -File tools/ba.ps1 smoke research-dtr-r0
```

The Codex desktop environment exposes the same setup and common actions through
`.codex/environments/environment.toml`.

## Android

Run Gradle only through the repository wrapper:

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDebug
```

Module boundaries are stable: `:app` owns the shell and assets,
`:feature:assist` runtime coordination, `:core:assist` pure risk logic,
`:core:vision` detection, `:core:device` Android adapters, and `:core:ui` state
and rendering.

## Evidence boundary

Synthetic, replay, curated Development, device, and natural evidence are named
separately. `UNKNOWN` and `NOT_EVALUABLE` are not negative evidence. A focused
demo result is never presented as universal real-world or safety performance.
Protected final evaluations use [formal research governance](docs/formal/RESEARCH_GOVERNANCE.md).

## License

See [CONTRIBUTING.md](CONTRIBUTING.md), [GOVERNANCE.md](GOVERNANCE.md),
[the model card](docs/MODEL_CARD.md), [maintainer automation](docs/operations/CODEX_MAINTAINER_AUTOMATION.md),
[the threat model](docs/THREAT_MODEL.md), [LICENSE](LICENSE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [SECURITY.md](SECURITY.md).
