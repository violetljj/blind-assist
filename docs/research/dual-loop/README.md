# BlindAssist 神经—几何双环研究主线

状态：`ISOLATED_ACTIVE_MECHANISM_LANDED / DEFAULT_OFF / FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`

最后核验：2026-07-31（Asia/Hong_Kong）

## 当前决定

2026-07-31 的 rank-2 Shiraz 设备评价已完成：baseline/candidate 均命中 `7/7`
正例，5 个 baseline-false 负窗全部保留，反馈行由 `508 -> 494`；终点为
`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`，不是事件级
误提醒改善。随后完成
[R1 事件失败分解](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)：
只消费三来源已关闭 Development trace、truth ledger 与 receipt，输出逐窗口字段、
retained-false 分类和不写 candidate trace 的 upper-bound audit。该 post-terminal
分析的唯一 top-level terminal 为 `POLICY_GRANULARITY_MISMATCH_SUPPORTED`；它不重写
R1 evidence、不改阈值、不授权或实现 R2。普通生产行为仍不变。

rank-1 仍按其自身终点 `FIRST_UNSEEN_SOURCE_NOT_EVALUABLE / VALID` 封存；它说明
truth-first 门在无足够正例时停止，不与 rank-2 的结果混为一谈。

2026-07-30 已完成
[隔离主动纠错 R1](DUAL_LOOP_ACTIVE_CORRECTION_R1_RESULT_2026-07-30.md)。
双环现已从真实 shadow 推进到独立 application id 的
`ACTIVE_CONTRADICT_ONLY`：几何环只在至少两个唯一关联框共同明显缩小时否决当前帧
提醒，其余全部弃权；raw/stable risk 与事件身份/生命周期规则不变，普通 debug、
release 和 shadow 仍默认不干预。该日的 CrowdBot/Matoaka 结果复现了触发行小幅下降
且已命中正例无新增延迟，但两个来源的负例提醒窗口均为 `7 -> 7`；当日 R1 终点是
`CROSS_SOURCE_ROW_SIGNAL_REPLICATED / NO_EVENT_LEVEL_EFFECT`。随后 Shiraz rank-2
与本次 failure decomposition 已将当前状态收口为顶部所列 frozen conclusion，均不构成
误提醒下降结论。

此前已完成
[因果框尺度三态源 R0](DUAL_LOOP_CAUSAL_TRACK_TRISTATE_R0_RESULT_2026-07-30.md)。
主线不再把 ego/target 运动责任归因、精确米制 TTC、pose、IMU、depth 或完整三维
恢复作为基础提醒的前置条件。当前第二环只对 production semantic loop 选中的目标，
根据连续 7 帧 `log(bbox height)` 的严格同号趋势输出
`CONFIRM_APPROACH / CONTRADICT_APPROACH / ABSTAIN`。

固定规则先在 8 个已烧毁 Development 会话上复现，再在任何选中 payload 打开前以
metadata-only hash 冻结 3 个全新 JRDB 会话、每个 360 帧。2D source 在 3D truth
取得前封存。独立 Confirmation 得到 1,017 个非弃权判断、1,008 个正确，总精度
`99.12%`；confirm `377/385 = 97.92%`，contradict
`631/632 = 99.84%`，coverage `2.391%`，三个会话均过预声明门。终点为
`ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS`，claim ceiling 仅为
`ANNOTATION_TRACK_MECHANISM_CONFIRMATION_ONLY`。

同一数学规则随后已进入 Android `AssistDecisionKernel` 的真实
`DUAL_LOOP_SHADOW` 路径：不再传入恒定 null，而是使用当前 production-selected
detection 与轻量 track continuity 生成三态 evidence，再由 frame/time/target/TTL
绑定的 admitter 准入。逐帧回归确认 risk、event、feedback、session 与 gateway
调用保持 baseline 完全一致。该 shadow 结果仍是主动纠错的工程前序，不被后继改写。

此前的
[真实几何 shadow cycle R0](DUAL_LOOP_REAL_GEOMETRY_SHADOW_CYCLE_R0_RESULT_2026-07-30.md)
仍作为工程前序保留：JRDB LiDAR 回放证明接缝可运行；Depth Anything temporal
derivative 与 homography residual flow 的负结果则解释了为何转向更简单、选择性
更强的框尺度三态源。

旧 Sparse LK F-1B 路线仍以 `NO_INCREMENT / VALID` 关闭；该结论没有被重写。
LITE R2 两臂均不达 readiness floor 后，主线不再优先把失败的 radial-flow 候选
接入 Android。当前最短可证伪路线改为对现有生产 `TemporalRiskTracker` 做
[factorial A/B R0](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_PROTOCOL_2026-07-30.json)：
A 只中和 object-detector temporal geometry output，B 保持当前完整生产链；每帧
QNN 只推理一次，两臂使用相同 detections 和完全隔离的决策/反馈状态。

[独立设计复核](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)
为 `PASS`。outcome-blind 预检已验证 `4422/4422` 个冻结 RGB；truth-membership
预检将 17 项原始 truth 冻结为 8 个可评分正例 + 7 个负窗，并在候选输出前排除
两个零有效帧正例。A/B factorization、truth-blind device producer、独立
validator/evaluator 已实现；核心回归、合成 mutation tests、Android build 均通过。
真机 prestart 在 `SM-S9280 / SM8650` 上复核 `4422/4422` 帧身份，并以 strict
QNN HTP 完成 synthetic probe，未解码 decision RGB、未写候选输出。hash-bound
implementation lock 与
[独立实现复核](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-30.md)
均已完成且为 `PASS`。唯一正式 producer、truth-blind validation/seal 与后续
truth evaluator 已完成；[执行结果](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_EXECUTION_RESULT_2026-07-30.md)
为 `VALID / NO_INCREMENT`。两臂在 8 个可评分正例、7 个负窗和两个 session 上的
实际提醒完全相同；正式 authority 已消费，Confirmation 不授权。

实现审计发现并已修复逐帧 timestamp 未绑定、truth receipt 未硬绑定、缺 post-validator
seal、review 未绑定当前 lock、host 并发启动窗口、producer marker 可覆盖、外部复制
pre-temporal hash 等正式前缺口。现在设备 producer 自身要求 lock + activation
authorization；validator 逐帧对照冻结 ledger 并发布哈希闭合 seal，evaluator 只接受
该 seal。修复后的独立 implementation 复审仍是 activation 的前置条件。

用户将双环设为新的研究主线后，2026-07-30 完成了独立 successor Discovery：
[可归因区域级接近证据源 Discovery R0](DUAL_LOOP_ATTRIBUTABLE_REGIONAL_APPROACH_SOURCE_DISCOVERY_R0_2026-07-30.md)。

LITE R2 负结果之后，用户已授权第一步冻结
[D0 ego-motion error attribution R0](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_PROTOCOL_2026-07-30.json)。
它只允许使用已经烧毁的 R2/REveL evidence，以 469 个 parent natural event 为
分析单位，描述性检查 person/sensor 径向分量、相机运动、ROI 抖动、事件长度、
flow MAD 与轨迹支持度。早先工程设计审查为
[`PASS / NOT_RUN`](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)，
但后续
[独立统计复核](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R0_STATISTICAL_REVIEW_RESULT_2026-07-30.md)
指出单 capture 依赖与“dominant mechanism”不可识别，终点为
`REPAIR_NEEDED / NOT_IMPLEMENTATION_READY / NOT_RUN`。D0 现作为生产 A/B
`NO_INCREMENT` 后的后备诊断，不再是唯一前瞻工作；旧 F-1B decision 继续密封。

本地只读连接复算确认，REveL Dynamic 的 RGB 人框、green/yellow 目标身份与
person/sensor Vicon 径向轨迹能在 LEFT/CENTER/RIGHT 全部区域输出目标可归因的
approaching/quasi-static/receding 开发真值。随后
[LITE R0 设计评审](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)
冻结完整连续 capture、两条最小 arm、输出/TTL/abstention、parent-event 分母与
停止规则并通过独立评审。两臂 producer、post-hash evaluator 和 24 个 synthetic
fixtures 随后完成并通过
[implementation review](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-30.md)；
一次性 activation 也通过独立复核。但唯一 R0 full producer attempt 在同目标相邻
RGB 尺寸从 `260×346` 变为 `258×346` 时触发 OpenCV LK 前提失败。按冻结规则
[LITE R0 execution result](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_EXECUTION_RESULT_2026-07-30.md)
为 `EXECUTION_INVALID_STOP_NO_RERUN / NOT_EVALUABLE`，未进入 truth join。

独立 R1 冻结跨尺寸处理后，正式 producer 完成，但共享 host guard 将 JSON 中 UTC `Z`
时间戳误解释为本地时间；R1 因执行包络门失败而同样停止，完整输出不作科学救援。
独立 R2 仅修复该执行包络、绑定新 identity/namespace，并通过设计、实现、pilot、
preflight 与 activation 评审。R2 的唯一 producer 和条件 evaluator 均有效完成。
[LITE R2 execution result](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R2_EXECUTION_RESULT_2026-07-30.md)
在冻结的 469 个 primary 自然事件上得到：

- box 面积增长：204/469 正确，153/469 wrong-signed；
- ROI 稀疏径向光流：188/469 正确，161/469 wrong-signed；
- flow 相对 box 的正确事件增量为 `-16`，两个 target 与三个区域增量均为负；
- 两臂均未达到正确率 `>=0.60`、wrong-signed `<=0.20` 的 readiness floor。

```text
PREDECESSOR_F1B: COMPLETE / NO_INCREMENT / VALID
SUCCESSOR_SOURCE_DISCOVERY: COMPLETE / SOURCE_FOUND_FOR_DEVELOPMENT
SUCCESSOR_LITE_DESIGN: DESIGN_REVIEW_PASS / F1_INTERFACE_FROZEN
RUNTIME_GEOMETRY_SOURCE: OFFLINE_IMPLEMENTED / ONE_SHOT_EVALUATED
SUCCESSOR_LITE_DEVELOPMENT: BOTH_NOT_READY_FOR_CONFIRMATION
CONFIRMATION: NOT_AUTHORIZED
EXECUTION_AUTHORITY: CONSUMED / NO_RERUN
CLAIM_CEILING: SINGLE_CAPTURE_ORACLE_ROI_CONDITIONED_DEVELOPMENT_ONLY
```

BlindAssist 的 predecessor 路线已按“先准入、后实现”顺序走完神经—几何双环
阶段−1，并在科学生死门停止：

```text
DATA_STATUS: READY
TIMING_STATUS: READY
SCIENCE_STATUS: NO_INCREMENT
RUNTIME_STATUS: NOT_RUN
ROUTE_CONTRACT_STATUS: MAINLINE_STOPPED
DATA_PROTOCOL_STATUS: VALID
TIMING_PROTOCOL_STATUS: VALID
SCIENCE_PROTOCOL_STATUS: VALID
RUNTIME_PROTOCOL_STATUS: NOT_RUN
EXECUTION_AUTHORITY: NONE
CLAIM_CEILING: DEVELOPMENT_ROUTE_REJECTION_ONLY
```

顺序固定为：

```text
F-1A 数据能否评价
  ↓
F-1B0 当前语义与几何结果何时真实可用
  ↓
F-1B 几何证据是否产生事件级增量
  ↓
F-1C 指定手机是否承载得住
  ↓
才决定是否实现正式双环
```

2026-07-30 的初始
[DUAL_LOOP_DATA_READINESS_R0](DUAL_LOOP_DATA_READINESS_R0_2026-07-30.md)
终点为 `HOLD_DATA`。经用户授权的固定既有 RGB 标签修复 R0 保持一次性终点不变；
独立后继 R1 只补缺失负类，最终达到
[F-1A `READY / VALID`](DUAL_LOOP_F1A_NEGATIVE_CATEGORY_SUPPLEMENT_R1_RESULT_2026-07-30.md)。

随后 [F-1B0 真机时序基线](DUAL_LOOP_F1B0_TIMING_BASELINE_R0_RESULT_2026-07-30.md)
在 `SM-S9280 / SM8650` 上形成生产 QNN 与隔离 Sparse LK 的完整
capture→available→consume 因果账本，终点为 `READY / VALID`。

F-1B 在 decision 候选输出仍为零访问时，对 hash-bound 的现有几何接口与生产提醒
状态机完成结构可达性检查。现有 Sparse LK 五通道只有全局中心走廊残差和质量，没有
目标身份、LEFT/CENTER/RIGHT 区域、接近方向、径向扩张或 TTC。在不伪造这些语义的
最薄融合下：

- 中心 `NEAR/CRITICAL` 在 A 分支已是 `HIGH`，一帧立即确认；
- 唯一有两帧确认延迟的可提醒分支是侧向 `NEAR/MEDIUM`，但全局中心走廊几何不能归因
  到 LEFT/RIGHT，必须 abstain；
- `MID/FAR/NO_CANDIDATE` 不得由残差升级距离、风险或创建提醒；
- 几何不得绕过既有 cooldown、fatigue 或实际交付语义。

因此 B 相对 A 的 `PAIRED_FIRST_DELIVERABLE_ALERT_LEAD` 理论上界为 `0 frame`。
[F-1B 结果](DUAL_LOOP_F1B_STRUCTURAL_REACHABILITY_PROTOCOL_REPAIR_R2_RESULT_2026-07-30.md) 为
`NO_INCREMENT / VALID`，按冻结合同停止论文双环主张，不消费 decision 集，也不进入
F-1C。

详细输入、状态、判定和停止门以
[双环阶段−1准入合同 R0](BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_CONTRACT_R0_2026-07-30.md)
为准。本轮执行来自用户连续推进授权；合同本身仍不构成未来新实验权限。

## 研究候选

- 语义证据：现有 YOLO11n 与现有生产检测/风险接口；目标设备若已准入，可使用其真实
  QNN 路由。
- predecessor 几何证据：Sparse LK 五通道已经证实不具备目标、区域和接近语义，
  保留为负结果与 regression fixture，不再作为 successor 的主候选。
- successor 几何证据：annotation-track `causal track tri-state` 保留为机制
  Confirmation；主动 R1 使用近期帧至少两个唯一关联检测框的共同缩小作为场景级
  `CONTRADICT`，否则弃权。两者均不使用运行时 truth、pose、IMU、depth 或米制 TTC。
- 汇合位置：同一事件/区域、真实时间戳、质量、时效和失效原因进入既有统一决策接缝；
  两个环不得分别提醒。
- 论文候选贡献：不是“双环”这个框图本身，而是双环相对 YOLO-only 是否产生可重复的
  首次有效提醒提前、风险判别改善或风险连续性改善。

运行时三态 source 已接入非干预 shadow；最小 scene-scale contradiction 已接入隔离
`dualLoopActive` 构建。当前不实现自适应调度、深度、分割、ARCore、新风险场、
latch、新状态机或第二套反馈系统。

## 与 RCLE、USTRF 和 Project Guideline 的关系

- [RCLE](../rcle/README.md) 已由用户于 2026-07-30 暂停。既有科学终态、协议终态、
  one-shot 消费状态和未消费的 `480+16` 全部保留；双环不重跑或救援 RCLE。
- Sparse LK 是阶段−1的默认轻量候选。RCLE 若未来恢复，只能通过新的、独立的准入决定
  成为可选证据源，不能自动替换 Sparse LK。
- 已关闭的
  [USTRF route-conditioned program](../ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)
  不因“双环”名称重启；不恢复旧 dense risk field、route、lifecycle 或旧数据门。
- [Project Guideline 适配审计](../../PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md)
  只提供失效语义、时间戳、最小证据账本和可重算原则；仍是
  `REFERENCE_ONLY / NO_IMPLEMENTATION_AUTHORITY`。

## 当前证据边界

本轮最终证据边界如下：

- 正式 CameraX 把 ImageAnalysis 配置为 `640×480 / KEEP_ONLY_LATEST`，Preview 请求
  `24 FPS`；该请求不等于真实 analysis 或结果频率。
- SM-S9280 / SM8650 的现有 QNN HTP 路由已经晋升，完整检测已有同机延迟与十分钟
  持续观测；能耗优势没有证据，SM8550 也没有被该结论覆盖。
- `LatestOnlySidecar` 与 `RgbaLumaSidecar` 已实现单槽替换、拥有的 luma 副本和过期
  结果拒绝，但不包含视觉算法、风险或提醒语义。
- 旧 CPU-era Sparse LK 回放与真机 shadow 结果相互提示：并发形态可能可承载，但真实
  CameraX 组合路径曾超过旧 `70 ms` 门，且没有 matched live YOLO-only 因果对照。
- 当前已有独立 annotation-track 三态方向证据、真实 Android shadow source、设备端
  4,422 + 10,724 + 4,891 帧 active replay 和隔离 APK smoke。CrowdBot、Matoaka 与
  Shiraz 三个 Development 来源只出现行级下降或持平，没有负例提醒窗口减少；普通
  生产行为因此保持不变。

因此 predecessor 只允许写成 `DEVELOPMENT_ROUTE_REJECTED / NO_INCREMENT`；
successor 当前只允许写成 `ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS /
ISOLATED_ACTIVE_MECHANISM_LANDED / DEFAULT_OFF /
FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`。
不得写成 live 算法有效、误提醒事件已下降、默认生产改善或安全结论。

## 当前权限

| 能力 | authority |
| --- | --- |
| 编写和维护阶段−1准入合同 | `AUTHORIZED` |
| F-1A 数据审计与既有 RGB 标签修复 | `COMPLETED / READY / VALID` |
| F-1B0 双源时间基线补测 | `COMPLETED / READY / VALID` |
| F-1B 几何增量评价 | `COMPLETED / NO_INCREMENT / VALID` |
| F-1B decision 输出执行 | `NOT_RUN / NOT_NEEDED / SEALED` |
| F-1C 手机双环 A/B | `STOPPED_BY_F-1B / NOT_AUTHORIZED` |
| successor 几何真值源 Discovery | `COMPLETED / SOURCE_FOUND_FOR_DEVELOPMENT` |
| successor causal runtime geometry | `ANDROID_SHADOW_IMPLEMENTED / ISOLATED_ACTIVE_IMPLEMENTED` |
| successor selective tri-state Development | `COMPLETE / CROSS_SESSION_REPLICATED` |
| production temporal geometry factorial A/B R0 | `COMPLETE / VALID / NO_INCREMENT / ONE_SHOT_CONSUMED` |
| D0 ego-motion error attribution | `R0_REPAIR_NEEDED / R1-R3_EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_R4` |
| successor 独立 Confirmation | `COMPLETE / ANNOTATION_TRACK_SOURCE_CONFIRMATION_PASS` |
| 一次有限修复 | `NOT_APPLICABLE_TO_MISSING_INFORMATION_SEMANTICS` |
| 第二环输入合同、准入门与生产影子接线 | `IMPLEMENTED / DEFAULT_OFF / SHADOW_NON_ACTUATING` |
| 隔离 active contradiction-only 构建 | `IMPLEMENTED / DEVELOPMENT_ONLY / NO_EVENT_EFFECT_CLAIM` |
| R1 event failure decomposition | `COMPLETE / POLICY_GRANULARITY_MISMATCH_SUPPORTED / DEVELOPMENT_ONLY` |
| scene-scale active successor / single-variable R2 | `CLOSED / NOT_WORTH_DESIGNING / NOT_IMPLEMENTED` |
| 默认生产 active/actuating 行为变更 | `NOT_AUTHORIZED` |
| 自适应调度、深度、分割、ARCore | `NOT_AUTHORIZED` |
| 默认模型、提醒、反馈或产品行为变更 | `NOT_AUTHORIZED` |
| 真人、独立助行、安全、产品或跨设备结论 | `NOT_AUTHORIZED` |

## 下一步

LITE R0/R1/R2、production temporal geometry factorial A/B R0 与 D0 R1/R2/R3
保持各自已消费的关闭终态，不重跑、不调阈值救援。隔离 active R1 已完成
[事件失败分解](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)：
row-density 仍可作为 Development diagnostic，但 25 个负窗中没有一个被完整消除。

upper-bound audit 只在内存中使用已记录的 R1 candidate opportunities：CrowdBot 与
Matoaka 各有一个满足既有正例命中、零 induced negative window 和预冻结新增时延上限的
有限 hold witness；它们都需要新的 runtime state，Shiraz 在 `250 ms` 上限内没有
witness。因此该 terminal 只支持解释 policy granularity，不授权任何 R2。

当前推荐关闭 scene-scale active 路线；不实现 hold、latch、事件状态、阈值调整或
单变量 R2。保留默认关闭的机制、receipt、回归夹具、逐窗口分解和失败分类；默认、
debug、release、产品、真人助行与安全行为均不改变。

### D0 R1/R2/R3 执行终态

[D0 R1 执行](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R1_EXECUTION_RESULT_2026-07-30.md)
因冻结解释器缺少 `rosbags`，在读取任何 bag message 前以
`EXECUTION_INVALID / CONSUMED / NO_SCIENTIFIC_EXIT` 关闭。

[D0 R2](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_PROTOCOL_2026-07-30.json)
随后只修复运行时包络，并通过设计复核、40 项测试、实现锁、独立实现复核与
activation。正式运行通过首条 Vicon message probe 后，在冻结 calibration parser
动态导入 `yaml` 时发现环境只有 `ruamel.yaml`。因此
[R2 执行结果](DUAL_LOOP_D0_EGOMOTION_ERROR_ATTRIBUTION_R2_EXECUTION_RESULT_2026-07-30.md)
同样为 `EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_SCIENTIFIC_EXIT`：
`0 / 469` event 完成、没有 event table、没有 D0 指标、没有科学出口。

R2 不能补包重跑。R3 以新的独立环境与 namespace 补齐 PyYAML 后，唯一正式运行
越过 runtime 问题，却暴露了冻结 D0 合同与上游 BBOX 字段之间的二倍语义冲突。
该 authority 已消费，协议与失败回执保持不可变；不生成 R4，不修补或重跑 R3。

## 2026-07-31：未见自然来源 rank-2 结果与 R1 收口

上海 rank-1 因 0 个正例在 baseline 前以 `FIRST_UNSEEN_SOURCE_NOT_EVALUABLE`
关闭后，按预冻结顺序启动 Shiraz rank-2。两路输出盲 RGB review 与第三路分歧裁决
冻结 7 个正例、6 个负窗，状态为 `TRUTH_FROZEN_ADEQUATE`。固定 4,891 帧 10 Hz
输入的 baseline-only adequacy 已通过，candidate 随后按一次性授权完成同一输入回放。

设备入口已拆为物理独立的 baseline-only 与 candidate-only。host 必须先确认
baseline 至少命中一个正例且误触发一个负窗，才生成绑定 input/truth/baseline SHA
的 candidate authorization；candidate 只重放同一 detections/metrics 并逐帧校验
raw/stable risk 不变。详见
[rank-2 protocol](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_PROTOCOL_2026-07-31.json)
和
[rank-2 truth result](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_TRUTH_RESULT_2026-07-31.md)。

“已使用”从此只限制同一候选的 unseen/independent claim，不对数据集作全局封存。
旧数据仍可作为 Development、回归或新问题来源；缺原生提醒标签的数据可在算法输出
打开前由隔离多模型复核补齐，但不能虚增独立 session。

rank-2 设备评价随后完成：baseline 和 candidate 都命中 7/7 正例，timely retention
为 7/7；但 5 个 baseline-false 负窗全部保留，`corrected=0`。全序列反馈行只从
508 降到 494。当前终点为
[`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`](DUAL_LOOP_R1_UNSEEN_NATURAL_EVENT_R0_RANK2_EFFECT_RESULT_2026-07-31.md)。
这关闭 active R1 的事件效果主张；默认生产保持关闭，shadow、机制结果、隔离 Android
工程和
[post-terminal failure decomposition](DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)
保留。后续不自动实现或设计 scene-scale active R2。

## 独立 successor 提案（proposal-only）

[TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0 设计合同](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_CONTRACT.md)
是一个不继承 R1 或 D0 权限的独立目标局部背景 warp residual 假设；经
[独立设计复核](TARGET_LOCAL_BACKGROUND_WARP_RESIDUAL_R0_DESIGN_REVIEW_RESULT_2026-07-31.md)
修订后，当前状态为 `DESIGN_REVIEW_PASS / PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`。
它只规定未来可审查的 B Development、
C1 新 session canary 与 C2 独立复现边界；不实现、不运行、不读取候选输出、不接 Android，
也不改变本 README 已冻结的 scene-scale active 关闭终点。
