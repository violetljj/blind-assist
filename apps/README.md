# Isolated Android apps and support modules

状态：`current / non-default-apps`

本目录收纳不属于默认 `:app` 的 Android 工程及其独立 canary 支撑模块。
模块注册与物理路径以 [settings.gradle.kts](../settings.gradle.kts) 为准；
部署、benchmark 或 canary 结果都不会自动影响默认 App。

| 分区 | 模块 | 职责 |
|---|---|---|
| `benchmarks/` | [`:device-benchmark`](benchmarks/device-benchmark/) | 设备评测 |
| `benchmarks/` | [`:ustrf-shadow-benchmark`](benchmarks/ustrf-shadow-benchmark/) | 隔离 shadow 研究 |
| `canaries/` | [`:hftf-device-canary`](canaries/hftf-device-canary/) | HFTF 真机 canary |
| `canaries/` | [`:hftf-metric-depth-canary-core`](canaries/hftf-metric-depth-canary-core/) | 度量深度 canary 的 JVM 逻辑与测试 |
| `demos/` | [`:hftf-depth-demo-app`](demos/hftf-depth-demo-app/) | 独立深度演示 |
| `demos/` | [`:known-height-capture-app`](demos/known-height-capture-app/README.md) | 已知高度采集 |
| `demos/` | [`:goal-capture-app`](demos/goal-capture-app/README.md) | 目标任务采集 |
| `demos/` | [`:semantic-anchor-demo-app`](demos/semantic-anchor-demo-app/) | 语义锚点演示 |
| `candidates/` | [`:npu-candidate`](candidates/npu-candidate/) | 隔离的 NPU 候选应用 |

规则：

- 默认产品实现只在根 `app/`、`core/`、`feature/`。
- 本目录模块不得被默认 `:app` 隐式依赖。
- 新实验 Android 工程必须进入对应分区，不得再次平铺到仓库根目录。
- 本地产物遵循[本地产物目录说明](../docs/LOCAL_ARTIFACTS.md)：普通 clone 使用模块 `build/`；配置外置 artifact junction 的工作区由 Gradle 重定向到 `artifacts.local/work/gradle-build/`。
