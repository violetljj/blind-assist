# 双环因果框尺度三态源 R0 结果

状态：`ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS / ANDROID_SHADOW_INTEGRATED`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

双环主线完成了一次方法论纠偏并落到真实代码：第二环不再把相机/目标运动责任归因、
精确米制 TTC、pose、IMU、depth 或完整三维恢复作为前置条件，而只对当前目标输出
`CONFIRM_APPROACH / CONTRADICT_APPROACH / ABSTAIN`。冻结候选在独立 JRDB
会话通过来源级 Confirmation，随后同一数学规则已接入 Android
`DUAL_LOOP_SHADOW` 的真实 decision kernel。

当前工程终点为：

```text
SEMANTIC_LOOP: production selected detection
CORRECTION_LOOP: causal seven-frame log-box-height tri-state
JOIN: DualLoopShadowAdmitter inside AssistDecisionKernel
RUNTIME: real Android shadow path
ACTUATION: forbidden
DEFAULT: OFF outside isolated dualLoopShadow build
```

这已经是端到端、可运行、会真实产生非空三态证据的双环影子落地，不再是空接口或
oracle-only host replay。但它还不是 active 提醒改进。

## 冻结规则

- 同一稳定 track 只看当前与过去共 7 帧；
- 对 `log(bbox height)` 随 capture timestamp 做因果 OLS；
- 6 次相邻变化全部为正且 slope `>= 0.2/s` 才确认接近；
- 6 次相邻变化全部为负且 slope `<= -0.2/s` 才否定接近；
- 其他情况一律弃权；
- 不读取 RGB、pose、IMU、depth 或 future frame。

完整门和来源见
[冻结协议](DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0_PROTOCOL_2026-07-30.json)。

## 独立 Confirmation

在任何选中标签 payload 打开前，先以 metadata-only hash 规则排除全部 13 个
outcome-open sequence，再冻结 3 个新 sequence、每个 360 个连续帧。随后只取得
2D box/track，封存 43,429 行 source 输出；此时 truth 路径不存在且 producer
receipt 记录 `truth_payload_opened=false`。最后才取得 3D truth 并评价。

| 指标 | 结果 |
| --- | ---: |
| 非弃权判断 | 1,017 |
| 总体正确 | 1,008 / 1,017 = 99.12% |
| `CONFIRM_APPROACH` | 377 / 385 = 97.92% |
| `CONTRADICT_APPROACH` | 631 / 632 = 99.84% |
| opportunity coverage | 2.391% |
| distinct tracks | 43 |

三个会话均通过预声明的 evidence、coverage 与 precision 门：

| sequence | evidence | coverage | precision |
| --- | ---: | ---: | ---: |
| gates-159-group-meeting-2019-04-03_0 | 54 | 2.827% | 92.59% |
| huang-lane-2019-02-12_0 | 698 | 3.476% | 99.28% |
| packard-poster-session-2019-03-20_2 | 265 | 1.290% | 100.00% |

Terminal 为 `ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS`。低 coverage 是有意的：
第二环宁可弃权，也不把噪声连续值伪装成逐帧纠错。

## Android 落地

`CausalTrackTristateGeometryProducer` 已进入 `core:assist`：

- 使用 production semantic loop 当前选中的 detection；
- 用 class/label/source、IoU 或归一化中心距离维持轻量 track epoch；
- target 切换、非单调帧或超过 500 ms 的相邻间隔会清空历史；
- 证据绑定 capture frame、availability、TTL、target 与 track epoch；
- kernel 仅在 `SHADOW_ABSTAIN_ONLY` 且调用者未提供 replay evidence 时运行；
- 普通 `DualLoopShadowAdmitter()` 仍为空 allowlist；kernel 只显式准入
  `CAUSAL_TRACK_TRISTATE_R0`；
- 隔离 runtime 以 `BlindAssistDualLoop` logcat tag 输出 frame、track epoch、
  disposition、三态、rate、quality 与 abstention reason，供下一门直接采集；
- admitted evidence 不能修改 risk、event、feedback 或 gateway 调用。

定向 Kotlin 回归覆盖一致增长、一致缩小、混合趋势弃权、target/gap reset，以及
七帧真实 kernel shadow admission。逐帧断言 baseline 与 shadow 的 raw/stable risk、
event、feedback、session summary 和 gateway call count 完全相同。

## 真机 smoke

在 `SM-S9280 / SM8650` 上安装独立包
`com.linnan.blindassist.dualloop.shadow`，冷启动 MainActivity 成功。授权该隔离包
相机权限并打开手机摄像头后，CameraX `480×640` 分析帧以连续 frame ID 到达
`BlindAssistDualLoop`；短观测中 detector/decision pipeline 约 24 FPS。镜头当时
没有检测到目标（`count=0`），因此影子源正确输出 `EVIDENCE_ABSENT`，未形成 live
非弃权三态样本。该 smoke 只证明真实相机帧身份、mode、kernel 与日志接通，不证明
live target continuity 或三态精度。

隔离包的 QNN route 因设备侧 `libcdsprpc.so` 对该安装环境不可见而回退
`cpu_xnnpack`；因此这次短观测不作为 NPU 或正式性能证据。smoke 后已 force-stop
隔离进程，APK 保留安装；正式 package 与其数据未修改。

## 边界与下一门

Confirmation 的 track identity 与 box 来自 JRDB annotation；Android 使用轻量
production-selected detection 连续性，两者不能偷换。因此当前允许主张：

> 简单因果框尺度三态机制已在独立 annotation tracks 上确认，并已完成真实 Android
> 非干预 shadow 集成。

当前不得主张 detector/live-track 精度、误提醒下降、提前提醒、漏报不增加、实时性能、
产品改善或安全效果。下一步不再增加算法模块；只需在 `dualLoopShadow` 真机路径收集
live 三态分布、track reset、延迟与 baseline parity。只有这些证据通过后，才讨论一个
范围很窄、可回滚的 active correction policy。
