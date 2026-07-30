# 双环隔离主动纠错 R1 结果

状态：
`ISOLATED_ACTIVE_MECHANISM_LANDED / DEFAULT_OFF / CROSS_SOURCE_ROW_SIGNAL_REPLICATED / NO_EVENT_LEVEL_EFFECT / NO_PRODUCT_OR_SAFETY_CLAIM`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

双环主线已经完成从真实 Android shadow 到隔离主动纠错的工程闭环，但没有获得可写成
“误提醒事件下降”的科学结果。R1 只在独立 application id 的
`dualLoopActive` 构建中允许几何环对当前帧执行 `CONTRADICT` 否决；普通 debug、
release 和 `dualLoopShadow` 均保持不干预。第二环不改 raw/stable risk、事件身份或
生命周期规则；事件的“已反馈”状态仍只按实际反馈是否被接受自然更新。

这条路线遵循任务充分原则：不恢复 ego/target 运动责任，不要求 pose、IMU、depth、
米制 TTC 或完整三维；只在至少两个唯一关联检测框共同表现为明显远离时，输出离散
`CONTRADICT_APPROACH`，其余情况全部 `ABSTAIN`。

## R0 否决与 R1 转向

完整 production detection dump 共 4,422 帧。原 R0 多轨迹三态得到：

- `ABSTAIN 3761 / CONFIRM 630 / CONTRADICT 20 / NO_TARGET 11`；
- 373 个 baseline 触发帧中只有 2 帧为 `CONTRADICT`；
- 可评分负例触发行中 `CONTRADICT = 0`。

因此 R0 终点为 `REJECT_MULTITRACK_ACTIVE_ROUTE`。它没有被接入主动路径。

R1 改用更小的场景尺度否决器：

1. 只在不超过 500 ms 的近期帧间对当前检测框做 greedy unique association；
2. 至少两个关联成功时，计算各框 `log(height)` 的瞬时变化率中位数；
3. 中位数不高于 `-0.05 / s` 才输出 `CONTRADICT_APPROACH`；
4. 无充分支持、信号冲突或不满足门槛时全部弃权。

这不是自运动估计，也不声称辨别“谁在运动”。它只是一个低成本的共同缩小反证。

## CrowdBot 开发回放

设备端使用真实 `AssistDecisionKernel` 对冻结的 4,422 帧完整检测结果逐帧运行 baseline
与 active：

- 全序列提醒触发行：`373 -> 357`；
- 可评分负例触发行：`27 -> 25`；
- 有提醒的负例窗口：`7 -> 7`，消除窗口数为 `0`；
- 正例召回：`8/8 -> 8/8`；
- 最大正例延迟：`0` 帧；
- raw/stable risk mismatch：`0` 行；直接 event mutation 权限始终为 `false`。

Host evaluator 与设备 Kotlin 实现逐帧对齐 `4422/4422`：detector hash、baseline
feedback、candidate feedback 和 scene decision mismatch 均为 `0`。设备 trace
SHA-256 为
`4b1d9896ebcca3571be158ba82c544f76c34e261428fe0efc4cef4a534be5994`。

终点：`ROW_BURDEN_SIGNAL_ONLY_NOT_WINDOW_EFFECT`。

## Matoaka 跨来源开发回放

第二来源使用完整 17:52 公共视频，固定为 10 Hz、640×480 letterbox，共
10,724 帧。设备 producer 只读取 RGB 与时间戳，未读取 truth；SM-S9280 上 strict
QNN HTP 完成运行：

- 全序列提醒触发行：`255 -> 247`；
- 可评分负例触发行：`51 -> 49`；
- 有提醒的负例窗口：`7 -> 7`；
- 正例召回：`3/7 -> 3/7`；
- 最大正例延迟：`0` 帧；
- risk mutation：`0` 行。

trace SHA-256 为
`fb66a9122ac293cfa2d94322684b012deadbaca8ab93108fb80cae75fad24462`。
Matoaka 标签来自此前在该视频上调过的 Development 标注，不是独立 Confirmation。
其合法终点仅为 `CROSS_SOURCE_DEVELOPMENT_SIGNAL_REPLICATED`。

## 复杂度停止门

已离线检查 100、200、300、500 ms 等否决保持时间：

- 100 ms 保持 CrowdBot `8/8` 正例，但负例提醒窗口仍为 `7`；
- 200 ms 时 CrowdBot 正例召回降为 `7/8`；
- 300 ms 时降为 `6/8`；
- 加入 selected-target shrink guard 仍未在保留正例的同时减少负例窗口。

因此 latch、新状态机或更长时序保持没有购买到事件级收益，且开始产生漏报风险。
R1 不实现这些复杂度。

## 工程落地

- `CausalSceneScaleTristateGeometryProducer` 实现最小离散反证；
- `DualLoopRuntimeMode.ACTIVE_CONTRADICT_ONLY` 仅允许 admitted contradiction
  抑制当前 planner-eligible feedback；
- `AssistDecisionKernel` 仍是唯一反馈接缝；第二环不改 raw/stable risk 或事件规则，
  只否决当前反馈机会；
- `dualLoopActive` 使用独立包名
  `com.linnan.blindassist.dualloop.active`，普通构建默认 `DUAL_LOOP_ACTIVE=false`；
- 主界面明确显示“神经—几何双环隔离纠错版 · 开发验证中 · 不可用于独立行走”；
- 真机已安装、冷启动并进入 `MainActivity`，无 fatal crash。

普通 APK 的 live smoke 因 app namespace 找不到 `libcdsprpc.so` 而回退至 CPU
XNNPACK。设备 benchmark 的 strict QNN HTP 回放证明了算法与设备执行链可运行，
但不能替代 active APK 的 live QNN 性能与持续运行证据。

## 最终证据边界

本轮已经完成“可运行、隔离、默认关闭、可逐帧反证、可真机启动”的主动纠错工程落地。
现有证据只支持跨来源 Development 的行级触发负担小幅下降；不支持误提醒事件下降、
正例覆盖改善、默认生产启用、真人助行、产品收益或安全主张。

CrowdBot 与 Matoaka 已烧毁，不再用于阈值、hold 或状态机调优。若未来要晋升效果主张，
最小新增证据应来自一个未参与设计的来源或无真人的脚本化 live 场景，并以事件级负例
减少、正例非劣和设备时延为联合门；这不是当前 R1 的未完成项。
