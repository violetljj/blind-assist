# Isolated Android apps

状态：`current / non-default-apps`

本目录收纳所有不属于默认 `:app` 的 Android 工程。Gradle module 名称保持不变，
仅物理路径分层；部署、benchmark 或 canary 结果都不会自动影响默认 App。

| 分区 | 模块 | 职责 |
|---|---|---|
| `benchmarks/` | `:device-benchmark`, `:ustrf-shadow-benchmark` | 设备评测和隔离 shadow 研究 |
| `canaries/` | `:hftf-device-canary`, `:hftf-metric-depth-canary-core` | HFTF 真机与 JVM canary |
| `demos/` | `:hftf-depth-demo-app`, `:known-height-capture-app` | 独立演示和采集应用 |
| `candidates/` | `:npu-candidate` | 隔离的 NPU 候选应用 |

规则：

- 默认产品实现只在根 `app/`、`core/`、`feature/`。
- 本目录模块不得被默认 `:app` 隐式依赖。
- 新实验 Android 工程必须进入对应分区，不得再次平铺到仓库根目录。
- 本地产物继续写入各模块 `build/` 或根 `artifacts.local/`，不得提交 Git。
