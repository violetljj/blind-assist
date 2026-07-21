# USTRF 跨相机连续事件工程 R1.2a（2026-07-21）

状态：**冻结协议已执行，事件门与设备门均失败；保持 benchmark-only，不消耗 R1.3 held-out。**

## 结论先行

R1.2a 把已经解封的 R1.1/R1.2 共 12 个视频全部降级为 `seen_diagnostic_not_held_out`，按冻结的 500ms 采样执行 5–15 秒连续重放。SM-S9280 完成 600 秒全速 soak，但总门未通过：正例事件召回仅 `4/6`，并且冻结 YOLOE-11s 640 的 inference latency 为 p50 `762ms`、p95 `978ms`，远高于 `<=120ms` 门。不得进入新的 held-out，不得替换默认模型。

## 连续事件结果

| 检查项 | 结果 | 解释 |
| --- | ---: | --- |
| 正例事件召回 | `4/6` | Edmonton、Thailand、Bridge、Roadwork 命中；Japan、London 在冻结 anchor 未匹配 |
| 首次正确告警 | `250/0/0/0ms` | 四个命中事件均在 `<=5000ms` 内；漏检事件不以此回救 |
| 负例目标假告警 | `0` | 五个 gate-eligible 负例均未交付目标告警；Vancouver 不入门 |
| 重复交付 | `0` | 37 次后续同事件尝试被抑制，没有第二次交付 |
| 共现物隔离 | `272` 个 route-inside 共现检测，`0` 次接管 | 普通共现物不能启动或维持指定目标事件 |
| 身份切换 | `0` | 已锚定 trace 未在后续冻结 anchor 暴露替换；association ambiguous rate `3.48%` |
| 目标出画 | 未过门 | Jakarta 已锚定并在 absent anchor 前清退，delay `0ms`；Japan/London/Ulm 因目标从未锚定而无法验证，fail closed，不用同类物替代 |

R1.1 的多 anchor 只用于复核同一 association trace；R1.2 单 anchor 的中间帧仍不是连续真值。`nearest_existing_frozen_anchor_polygon_held_constant` 只帮助工程诊断，不是米制几何或 route truth。

## SM-S9280 稳定性、延迟与温升

- 设备：`R5CX10M8Y8X / SM-S9280 / Android 16 API 36`。
- 完整 soak：`600s`；总 inference 样本 `648`；decode failure `0`，inference failure `0`，instrumentation 没有提前崩溃。
- inference latency：p50 `762ms`、p95 `978ms`；冻结门为 p95 `<=120ms`，明确失败。该数值不含“用热稳定掩盖性能不可用”的解释空间。
- 电池温度：`29.6°C -> 34.7°C`，最大增幅 `5.1°C`；冻结门分别为 `<=45°C` 与 delta `<=8°C`，通过。
- Android thermal status 全程最大为 `0`，通过 `<=2` 门。

因此，连续运行的崩溃/温升风险在本次压力窗口内没有暴露，但当前静态三类 YOLOE Android 执行速度不满足连续助行调度，设备总门仍为 false。

## R1.3 预注册与访问边界

R1.3 已在 R1.2a 协议/门槛冻结后预注册为 12 个尚未打开的来源槽位：`6 positive + 6 negative`，每个父来源只允许一个 5–15 秒事件。当前没有发现、下载、解码或运行任何新来源，`result_access_authorized=false`。

event truth 冻结为两次独立 fresh-context 大视觉模型复核：reviewer 不得看到 detector 输出，必须复核物理目标身份、可见区间、alertable start、passed/cleared、路线关系、共现排除和 should-alert；身份/关系/事件标签要求完全一致，时间允许 `500ms`，冲突必须第三次独立裁决。其权威仅为 `dual_vlm_reviewed_provisional_event_truth`，不是 human truth 或生产授权。

Vancouver 固定为下一轮漏检 taxonomy 线索，不能改变 prompt、类别、`.05/.30`、bbox、polygon、R1.2a 门槛或 R1.3 truth。

## 决策与证据

- 决策：`do_not_replace_default_model`；R1.3 inventory 保持锁定。
- 下一步只能在这 12 个 seen diagnostic 上解决 Android 延迟与 Japan/London anchor miss，并按同一 R1.2a 协议复跑；在 p95、6/6 事件和出画证据闭合前，不解封新 held-out。
- 协议：`configs/ustrf_crosscam_continuous_events_r12a_seen_v1.json`，SHA-256 `57bf4bea498af6c0817e4ee595c546760418cd7e0acab1dc97835e163555fd49`。
- R1.3 预注册：`configs/ustrf_crosscam_continuous_events_r13_prereg_v1.json`，SHA-256 `4ad7a29bc98693e16a7567e1bdcd9b0f1b71d780e56e511f5b394da09b7965c5`。
- Android 输出：`artifacts.local/evidence/ustrf-crosscam-codex/continuous-r12a-seen-diagnostic-v1/android-arm/android_r12a_continuous_output.json`，SHA-256 `8a7b23aa2fae88bb735f561ed0f5378f8782eac76b610e4a5b89c95eb3385173`。
- 设备回执：`android-arm/device_run_receipt.json`，SHA-256 `aa266f76a7696dc7159ea99a61f5fe97b4b3aa7ba706828698d06072ec3a99d0`。
- 证据索引：`evidence_index.json`，SHA-256 `31101ef04e8f4646c8b02185ac264bdd8185ad1bb688f225d270dc87e355e92a`。
