# BlindAssist 三分钟快速开始

状态：`current`

最后核验：2026-08-13

本页让新贡献者从克隆仓库走到一个真实、可复核的结果。三分钟覆盖定位和启动专项检查；
首次 Android 构建若需下载 Gradle 依赖，可能更久。

> BlindAssist 是研究与无障碍原型，不是经过认证的助行或安全设备。构建成功不证明
> 感知质量、用户效果或安全导航能力。

## 开始前

准备 Git、PowerShell 7、JDK 17 与 Android SDK Platform 35。Windows 路径由仓库脚本
统一发现并校验工具链；本快速开始不需要数据集、研究 payload、真机或任何密钥。

## 0:00 — 克隆并进入仓库

```powershell
git clone https://github.com/violetljj/blind-assist.git
cd blind-assist
git status --short --branch
```

预期看到 `master` 且没有本地改动。

## 0:45 — 选择最小有效路径

### 文档或社区贡献

```powershell
pwsh -NoProfile -File scripts/check_docs_index.ps1
```

这项检查不需要 Android SDK 或设备。开源治理材料变化时，再运行其直接负责的
`scripts/check_open_source_readiness.ps1`；不要为普通文档修改默认执行无关全仓门禁。

### Windows Android 贡献

先做有界环境预检：

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 -PreflightOnly
```

若返回 `ENV_BLOCKED`，按输出修复 JDK 或 SDK；不要绕过统一 wrapper。

## 2:00 — 产出一个可证伪结果

只运行覆盖本次改动的最小任务。例如修改 `:app` 时：

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:testDebugUnitTest
```

需要验证 APK 装配时再运行：

```powershell
pwsh -NoProfile -File scripts/run_android_gradle.ps1 :app:assembleDebug
```

输出位于 `app/build/outputs/apk/debug/app-debug.apk`。纯文档修改通常只需：

```powershell
git diff --check
pwsh -NoProfile -File scripts/check_docs_index.ps1
```

结构、发布、权限、默认 App 或共享基础设施变化应按对应 owning 文档提高验证强度，
但不因“准备提交”重复运行无关检查。

## 3:00 — 选择一个有边界的首次贡献

- 从 [`good first issue` 列表](https://github.com/violetljj/blind-assist/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)开始；
- 开始前在 issue 留言，避免重复劳动；
- 只改 issue 声明的文件，运行其中可复制的专项验证命令；
- 外部贡献者提交一个聚焦 PR，报告命令、结果、剩余缺口以及是否影响默认 App。

编辑前阅读 [CONTRIBUTING.md](../CONTRIBUTING.md)。安全或隐私问题按
[SECURITY.md](../SECURITY.md) 私下报告；一般问题和早期想法可进入
[GitHub Discussions](https://github.com/violetljj/blind-assist/discussions)。

## 首次贡献不要做什么

- 未获明确范围时，不修改打包模型、权限、反馈策略、发布签名、研究门或默认 App authority；
- 不提交原始相机素材、设备日志、私人数据、凭据、受限数据集、SDK payload、APK 或
  `artifacts.local/`；
- 不把 `UNKNOWN`、synthetic evidence、构建成功或 benchmark 写成产品或安全证明。

稳定模块定位见[代码地图](CODE_MAP.md)，完整证据边界见[研究治理](RESEARCH_GOVERNANCE.md)。
