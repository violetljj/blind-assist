# HFTF Stage C D34：Kotlin shadow-state parity/runtime result

日期：2026-08-03

证据角色：Development / production Kotlin mechanism parity and host runtime

研究主线：不变

默认 App：不变

## 结论

D34 把 D33 全部 5,366 个 source-only detector-track occurrences 交给当前生产
Kotlin `CausalTrackTristateGeometryProducer`，逐条与冻结 Python source rule
比较。全部 parity 与 runtime gates 通过：

`D34_KOTLIN_SHADOW_STATE_PARITY_RUNTIME_SUPPORTED`

当前建立：

`PRODUCTION_KOTLIN_CAUSAL_TRACK_STATE_PARITY_AND_HOST_RUNTIME_SUPPORTED`

这说明 D33 的正机制已经不只是离线 Python 结果；仓库中实际用于 isolated shadow
路径的 Kotlin decision implementation，在真实 detector tracks 上没有 decision
语义漂移，计算成本也远低于当前 host gate。

## 结果

| metric | result | gate |
|---|---:|---:|
| corpus rows | 5,366 | = 5,366 |
| detector tracks | 165 | descriptive |
| decision mismatches | 0 | = 0 |
| slope presence mismatches | 0 | = 0 |
| max absolute slope error | `8.44e-7 / s` | <= `1e-5 / s` |
| producer call P50 | `0.0014 ms` | descriptive |
| producer call P95 | `0.0022 ms` | <= `0.10 ms` |
| producer call P99 | `0.0044 ms` | descriptive |
| `core:assist` tests | PASS | PASS |

P95 只占冻结上限的 2.2%；即使 host timing 不能替代设备测量，也足以排除
tri-state OLS 本身是明显 runtime bottleneck。

## parity corpus

Python materializer 只读取：

- D33 `tracks.jsonl`；
- 四个 packet 的 frame timestamp；
- bbox/track identity；
- 冻结七帧 source rule。

它不读取 D33 的 current annotation matches、native identities、3D range 或 future
truth。track frame gap 会显式 reset，与在线缺失目标时的 history reset 语义一致。

- input rows：5,366；
- distinct tracks：165；
- input SHA-256：
  `d1f24dc7c61890e912d2a4a1cbca23e4b729dfceb1ef76b435cd573c97e6021e`；
- input receipt SHA-256：
  `0fd913e1a1d20264b549c2498ddec88d997e1609d57c64a0aec771895f6f89c7`。

## Kotlin execution

- 直接使用 production class，没有复制一份 Kotlin reference；
- 每个 detector track 独立 producer；
- 第一遍完整 warm-up，第二遍逐 call 计时；
- 使用 `REPLAY_TIMELINE` clock domain；
- 不进入 `AssistDecisionKernel`、event tracker、feedback planner 或 gateway；
- `non_actuating=true`、`future_truth_consumed=false`。

report：

- size：610 bytes；
- SHA-256：
  `c6ac570f19cf5d06f00dc159b920f75dbbd44be1d2808949bc894620631a9247`。

第一次全量测试命令在编译前因 PowerShell 未引用 `-Dorg.gradle.jvmargs`，被 Gradle
误读为 task name；修正命令行引用后同一 `core:assist` 测试成功。这是工程命令错误，
没有产生 D34 终态，也没有消耗或关闭 corpus。

## 证据边界

D34 建立 production Kotlin source-rule parity 与 host JVM 计算成本，但仍不建立：

- 物理 Android 设备 latency；
- CameraX frame/track continuity；
- on-device detector + state end-to-end coverage；
- event utility、提醒增量、默认 App、产品效果或 human safety。

这些是下一步而不是对 D34 正结果的否定。

## 下一步

D35 进入 isolated `dualLoopShadow` 物理设备 canary：

1. build/install 独立 `.dualloop.shadow` variant；
2. 保持 production/default flags 与 alert path 不变；
3. 用 bounded replay 或 camera source 采集 shadow state census；
4. 测量 device producer P50/P95、frame gaps、history resets、non-abstain coverage；
5. 核对 baseline risk/event/feedback trace 与 shadow-off 完全一致；
6. 只有 device parity/non-interference 通过，才进入 event-level shadow utility
   comparison。

支线仍不替换传统主线；未来只有 event-level utility 在独立数据上超过主线，才具备
主线晋级候选资格。
