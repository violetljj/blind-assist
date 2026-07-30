# BlindAssist 神经—几何双环研究主线

状态：`successor Development / IMPLEMENTATION_REVIEW_PASS / REPLAY_NOT_RUN`

最后核验：2026-07-30（Asia/Hong_Kong）

## 当前决定

旧 Sparse LK F-1B 路线仍以 `NO_INCREMENT / VALID` 关闭；该结论没有被重写。
用户将双环设为新的研究主线后，2026-07-30 完成了独立 successor Discovery：
[可归因区域级接近证据源 Discovery R0](DUAL_LOOP_ATTRIBUTABLE_REGIONAL_APPROACH_SOURCE_DISCOVERY_R0_2026-07-30.md)。

本地只读连接复算确认，REveL Dynamic 的 RGB 人框、green/yellow 目标身份与
person/sensor Vicon 径向轨迹能在 LEFT/CENTER/RIGHT 全部区域输出目标可归因的
approaching/quasi-static/receding 开发真值。随后
[LITE R0 设计评审](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_DESIGN_REVIEW_RESULT_2026-07-30.md)
冻结完整连续 capture、两条最小 arm、输出/TTL/abstention、parent-event 分母与
停止规则并通过独立评审。两臂 producer、post-hash evaluator 和 24 个 synthetic
fixtures 随后完成并通过
[implementation review](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-30.md)；
全量 replay 仍未运行：

```text
PREDECESSOR_F1B: COMPLETE / NO_INCREMENT / VALID
SUCCESSOR_SOURCE_DISCOVERY: COMPLETE / SOURCE_FOUND_FOR_DEVELOPMENT
SUCCESSOR_LITE_DESIGN: DESIGN_REVIEW_PASS / F1_INTERFACE_FROZEN
RUNTIME_GEOMETRY_SOURCE: IMPLEMENTATION_REVIEW_PASS / REPLAY_NOT_RUN
CONFIRMATION: NOT_AUTHORIZED
EXECUTION_AUTHORITY: SEPARATE_ONE_SHOT_ACTIVATION_DECISION_REQUIRED
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
- successor 几何证据：候选为 `target/track-conditioned causal radial geometry`，
  仅使用过去帧，在同一目标 ROI 内计算框面积增长和/或稀疏径向光流；REveL Vicon
  只提供离线开发真值，不是已实现的运行时输入。
- 汇合位置：同一事件/区域、真实时间戳、质量、时效和失效原因进入既有统一决策接缝；
  两个环不得分别提醒。
- 论文候选贡献：不是“双环”这个框图本身，而是双环相对 YOLO-only 是否产生可重复的
  首次有效提醒提前、风险判别改善或风险连续性改善。

当前不自动实现运行时几何源、正式融合器、自适应调度、深度、分割、ARCore、新风险
场、新状态机或第二套反馈系统。

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
- 当前不仅没有双环事件增量证据；在既有五通道 Sparse LK 与不改变提醒语义的约束下，
  还不存在能改变实际可交付提醒的合法作用路径。

因此 predecessor 只允许写成 `DEVELOPMENT_ROUTE_REJECTED / NO_INCREMENT`；
successor 只允许写成
`IMPLEMENTATION_REVIEW_PASS / REPLAY_NOT_RUN / RUNTIME_SOURCE_NOT_EVALUATED`。
不得写成算法有效、双环有效、早提醒已实现、运行时可行、产品改善或安全结论。

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
| successor causal runtime geometry | `IMPLEMENTATION_REVIEW_PASS / REPLAY_NOT_RUN` |
| successor Development round | `IMPLEMENTATION_REVIEW_PASS / ACTIVATION_DECISION_REQUIRED` |
| successor 独立 Confirmation | `NOT_AUTHORIZED` |
| 一次有限修复 | `NOT_APPLICABLE_TO_MISSING_INFORMATION_SEMANTICS` |
| 正式融合器或生产 CameraX 接线 | `NOT_AUTHORIZED` |
| 自适应调度、深度、分割、ARCore | `NOT_AUTHORIZED` |
| 默认模型、提醒、反馈或产品行为变更 | `NOT_AUTHORIZED` |
| 真人、独立助行、安全、产品或跨设备结论 | `NOT_AUTHORIZED` |

## 下一步

LITE R0 implementation identity 已冻结并通过独立评审。当前唯一后继动作是另立、
绑定当前 implementation commit/lock/input/environment/output 的一次性 activation
decision 并独立复核；该 decision 通过前不得执行全量 producer replay 或 truth join。

旧 F-1B sealed decision 集不得用于回调规则，旧真机 timing-only 凭据也不得改写成
效果证据。新的独立 Confirmation 必须在候选输出访问前另行冻结，当前没有自动执行
权限。
