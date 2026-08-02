# HFTF Stage C D34：Kotlin shadow-state parity/runtime canary

日期：2026-08-03

证据角色：Development / production Kotlin mechanism parity and host runtime

研究主线：不变

默认 App：不变

## 问题

D33 已证明 offline YOLO11n + ByteTrack 的七帧 causal state 能高精度预测一秒
future range。D34 不再读取 future truth，也不训练或调整任何模型。它回答：

> 当前 `core:assist` 生产 Kotlin
> `CausalTrackTristateGeometryProducer`，对 D33 的真实 detector tracks 是否与
> 冻结 Python source rule 逐 occurrence 一致，并且计算成本足够低？

## 输入与 reference

- 输入：D33 `tracks.jsonl` 的全部 5,366 source-only track occurrences；
- timestamp：四个既有 JRDB packets 的 `image_timestamp_ns`；
- reference：D32/D33 已冻结的七帧 contiguous
  `log(box_height)` OLS tri-state；
- gap：track frame index 不连续时显式 reset；
- 不携带 2D annotation match、native identity、3D range 或 future outcome。

Python materializer 输出 deterministic TSV：

`sequence / track_id / frame_index / timestamp_ns / bbox /
expected_decision / expected_slope`

## Kotlin execution

- 直接调用生产类 `CausalTrackTristateGeometryProducer`；
- 每个 D33 detector track 独立 producer；
- frame gap 显式 `reset()`；
- 第一遍完整 corpus warm-up；
- 第二遍逐 occurrence 计时并核对 decision/slope；
- 不调用 `AssistDecisionKernel`、event tracker、feedback planner 或 gateway；
- 不改变 build flags、默认模式、alert、UI 或 runtime route。

## gate

1. corpus rows = 5,366；
2. decision mismatch = 0；
3. null/non-null slope mismatch = 0；
4. max absolute slope error <= `1e-5 / s`；
5. host JVM producer-call P95 <= `0.10 ms`；
6. `:core:assist:test` 通过。

通过：

`D34_KOTLIN_SHADOW_STATE_PARITY_RUNTIME_SUPPORTED`

未通过：

`D34_KOTLIN_SHADOW_STATE_PARITY_RUNTIME_NOT_SUPPORTED`

路径、Gradle、JDK、parser 或写出失败是 engineering failure，修复后重跑；不产生
科学负终态，也不关闭 D33。

## 主张边界

通过只建立 production Kotlin decision semantics 与 D33 Python source rule 的
一致性，以及 host JVM 上的计算成本。它不是物理设备 latency、CameraX integration、
tracker continuity、event utility、默认 App、产品效果或 human safety 证据。

通过后下一步才是 isolated `dualLoopShadow` build 的物理设备 replay；仍不驱动提醒。
