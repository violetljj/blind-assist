# HFTF 候选支线章程 R0

日期：2026-08-01

workflow：`DEVELOPMENT_STANDARD`

阶段：`REVERSIBLE_EXPLORATION`

状态：`CANDIDATE_SIDE_LANE_ACTIVE / MAINLINE_UNCHANGED`

## 1. 决策

建立 HFTF 独立候选支线。支线可以使用自己的数据、teacher、student、评价与失败终态，
但不修改当前双环主线的协议、证据或终态。只有通过第 7 节的公平 challenger gate，
HFTF 才能成为新的**研究主线**；正式 App、主动提醒与生产默认值不随该决策自动变化。

## 2. 科学问题

在只允许因果历史 RGB 输入的条件下，轻量 student 能否预测面向行人身体包络的短时未来
可通行/碰撞风险场，并在 parent-event utility 与设备成本上超过届时的研究主线
incumbent？

该问题分成三个不可混写的层次：

1. **teacher mechanics**：metric geometry 能否稳定生成多高度、短时未来 proxy；
2. **student prediction**：历史 RGB 是否包含超过单帧/current-field 的预测增量；
3. **decision utility**：在相同 canonical decision kernel 与输出语义下，各 arm 使用
   outcome 前冻结的 source adapter，预测场是否改善事件级效用。

上一级通过不等于下一级通过。

## 3. 输出合同

最小语义张量为：

`F[theta_bin, distance_bin, horizon, height_band]`

每个 cell 至少输出：

- `risk_score`：`[0, 1]`，表示几何代理下的人形扫掠冲突强度；
- `known_score`：`[0, 1]`，表示该 cell 的 teacher/student 可判定度；
- `provenance`：teacher/source/evidence version；
- 可选 tri-state：只能由 outcome 前冻结阈值导出
  `SAFE / RISK / UNKNOWN`。

`height_band` 的稳定语义是 `foot / body / head`；它描述身体占据空间，不是语义类别。
`horizon` 必须显式包含 current；future 点的具体数量和时间边界在 H1 outcome 前冻结。
`theta`、distance bin、身体尺寸、known/risk threshold 也必须在对应 protocol 中冻结，
本章程不凭空给出数值。

核心场同时评价全部候选方向，不从 RGB 声称推断用户意图。路线选择、语音、震动和
STOP/LEFT/RIGHT policy 均是下游 adapter，不进入 H0/H1 representation claim。

`theta / distance / height`、人体包络、head/dynamic/uncertainty 都继承自历史 USTRF
primitive，不构成本支线的新颖性主张。HFTF 要证明的是 history-only RGB 对显式
short-future layered cells 的表示增量。

## 4. Teacher 与 student 防泄漏

- teacher 可读取 metric depth、相机内外参、future frames 和经验证的 pose-frame
  binding；
- student 只能读取决策时刻及之前的 RGB/内部状态；
- teacher 未来信息不得进入 student 输入、阈值选择或测试时 adapter；
- dynamic object、遮挡、pose 质量和 body/ground transform 不充分时必须输出
  `UNKNOWN`，不得默认当作 `SAFE`；
- teacher proxy agreement 只能证明蒸馏 fidelity，不能证明真实风险或用户效果。

## 5. 与近邻和历史路线的因果差异

历史 USTRF route-conditioned program 已关闭。HFTF 不重开其 dense/bbox-route、
tracker/TTC/lifecycle 或已消费 15-pair selection。`g=(theta,rho,z)`、地面/腰部/头部、
身体包络、dynamic 与 uncertainty 均按继承 primitive 处理；最小待证差异只限
action-agnostic、history-RGB、显式 short-future layered-cell prediction 及其相对
current-only 的表示增量，并把 action policy 后置。

外部近邻要求直接比较：

| 近邻 | 已覆盖内容 | HFTF 尚待证明的差异 |
| --- | --- | --- |
| [AgniNav](https://arxiv.org/abs/2606.10903) | 身体碰撞包络、几何/深度标签、单目 student、极坐标 bin、边缘端 | BLV 行人任务、foot/body/head 分层、短时动态场、手机与弃权 |
| [AI Guide Dog](https://arxiv.org/abs/2501.07957) | 1 秒未来方向预测、iPhone/CoreML | 碰撞风险而非未来行进方向；多高度与 unknown |
| [Capability-Aware Traversability](https://arxiv.org/abs/2607.20679) | embodiment-conditioned traversability | 人体分层、时域与助盲事件效用 |
| [EgoWalk](https://arxiv.org/abs/2505.21282) | 胸前 RGB/depth/odometry 与 future-path 弱标签 | 非机器人启发式真值；头部/动态风险与 BLV 事件 |
| [Running Guide Agent](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/running-guide-agent/) | 端侧快速分割 + 低频语义推理 | HFTF 张量与可复核系统评价 |
| [EgoNav](https://arxiv.org/abs/2403.19026) / [Egocentric Future Localization](https://openaccess.thecvf.com/content_cvpr_2016/html/Park_Egocentric_Future_Localization_CVPR_2016_paper.html) | 第一视角未来位置/路径预测 | layered collision field、unknown 与助盲事件效用 |
| [Navigation World Models](https://arxiv.org/abs/2412.03572) / [NavWM](https://arxiv.org/abs/2606.24101) | action-conditioned future observation / navigation world modeling | HFTF 不是 world model；只比较任务特定短时风险预测 |

因此当前只允许“任务与表示组合假设”，不允许“首次发明身体条件可通行场”。

## 6. 分阶段实验

### H0 — Source/teacher feasibility

- 输入：hash-bound RGB、metric depth、intrinsics、pose、body/ground contract；
- 输出：静态投影、多高度 teacher、future teacher、effect-evaluation 四级裁决；
- 当前终态：`HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`；
- 通用 H0 只可准入静态投影；pose/body sidecar 即使结构与 hash 全部通过，也不能自行
  认证 source-native frame/time mapping 或物理标定；
- H1 前置已完成：source-specific verifier 从 official loader/GCS receipt 复算
  pose-frame mapping，并在一个 discovery + 三个独立 replication sessions 上冻结并
  复现 metric-depth transform 与 per-frame local-ground proxy；
- H1 body center 只可使用 camera world position 到 source-derived local ground plane
  的正交投影；它不是 physical camera-to-person calibration。若 H1/H2/H3 要提出真实
  人体碰撞或 participant claim，缺失物理标定仍必须 `NOT_EVALUABLE`；
- 停止：任何绑定或权威缺失即在相应层 `NOT_EVALUABLE`，不靠默认索引或自报 provenance
  补齐。

### H1 — Geometry teacher canary

- 当前授权：`H1_GEOMETRY_TEACHER_CANARY`，尚未执行；
- 比较：current single-height、current multi-height、future multi-height；
- 独立单元：parent source-session，不把 frames 当独立样本；
- 必须报告：reprojection/pose validity、unknown coverage、各高度层冲突一致性、
  dynamic/occlusion failure atlas；
- 成功只允许：`GEOMETRY_PROXY_MECHANISM_SUPPORTED`；
- 失败关闭：精确定义 teacher/evidence version，不外推整个问题。

### H2 — Causal student

- 同 backbone/预算比较 single-frame 与 history-only temporal student；
- train/dev/test 按 parent session/source 隔离；
- teacher fidelity 与独立 event utility 分表报告；
- 直接基线至少包括 single-frame、history current-only、历史 USTRF primitive-compatible
  表示，以及适用的 future-path predictor；
- 若 future 轴不能超过 current-only，则关闭该 future formulation，不用最好片段救援。

### H3 — Incumbent challenge

- 使用相同 parent-event cohort、decision clock、canonical decision kernel、输出语义
  与设备 harness；各 arm 的 source adapter 可不同，但必须预冻结并计入该 arm；
- arms 至少包括正式 YOLO baseline、届时研究主线 incumbent、HFTF；
- outcome 前冻结 operating point、non-inferiority margin、missing/abstention 规则；
- H3 在获得执行授权前必须用 precision/power 目标冻结最少独立 source/ancestry、
  parent-event 与 positive/negative/critical 分层单元；所有最小数都必须在
  source-native ancestry 去重后达到，未冻结或未达到时 `NOT_EVALUABLE`；
- 每个 parent event 必须绑定 source-native `raw_capture_ancestry_id`、
  `source_recording_id`、原生起止时间、asset generation/checksum 与
  `parent_natural_event_id`；这些键由独立 validator 从来源 inventory 重算，候选自报的
  event/session/source alias 不能增加独立单元；
- 同一 raw capture 或时间重叠的派生事件在所有 arms/strata 中只计一次；无法判定来源
  权威时终止为 `H3_EVENT_IDENTITY_AUTHORITY_NOT_EVALUABLE`，发现重复/重叠未解决时终止为
  `H3_DUPLICATE_OR_OVERLAPPING_EVENT_NOT_EVALUABLE`；
- 独立 validator 从 source ledger 重算，不导入 candidate decision code。

## 7. 研究主线晋级门

co-primary：

1. `critical_hazard_parent_event_recall`
2. `first_valid_warning_lead_time`

guardrails：

- false alert parent events / active minute；
- wrong-direction recommendation rate；
- passed/clear correctness；
- `UNKNOWN` coverage 与错误落入 `SAFE` 的比例；
- minimum source/session result；
- P50/P95 latency、peak memory、energy/thermal 可用指标。

晋级要求：

1. 在 reserved parent-session/source-held-out cohort 上，至少一个 co-primary 超过
   incumbent 的预冻结 superiority margin；
2. 另一个 co-primary、全部安全相关 guardrail、最差 source 和设备预算均满足预冻结
   non-inferiority/upper-bound；
3. 增量在新的 source/ancestry 与独立 parent events 上复现；同一来源只换 evidence
   version 只能算 regression，不能获得 replication 信用；
4. 通过不导入 candidate 的独立 validator；
5. 不以 teacher agreement、frame-level 数量、可视化热图或平均值掩盖 event/worst-group
   失败。

所有预准入 natural parent event、decision-required tick、passed/clear opportunity、
active exposure interval、device attempt 与 required field cell 都保留在各自预冻结
指标分母中；HFTF 的 `UNKNOWN`、abstention、timeout、invalid 或 missing 输出不能把
单元移出分母。只有对全部 arms 共同适用、在 outcome 前冻结的 source-level 不可评规则
才允许排除。

逐 guardrail 的分母与 missing/censoring 记分也必须 outcome 前冻结：

- false alerts 使用全部预准入 negative/non-critical active exposure；
- wrong direction 使用全部 direction-required decision ticks，弃权/缺失按预冻结
  error/censoring 规则计分；
- passed/clear 使用全部对应 opportunity，无输出不得消失；
- unknown coverage 使用全部 required cells/ticks，包括缺失 field；
- minimum-source 使用 ancestry 去重后的全部预准入 natural events；
- latency/memory/energy/thermal 使用全部 scheduled device attempts，timeout、crash、
  invalid 与 thermal abort 按预冻结 failure/worst-case 规则计分。

superiority/non-inferiority 必须报告预冻结的置信界或等价有界不确定性，而不是只比较
点估计。missed critical event 不能从 lead-time 分布删除；H3 protocol 必须在读取
outcome 前冻结其 worst-case/censoring 记分规则。incumbent model、source adapter、
decision kernel、cohort、protocol 与 validator 均绑定精确 hash。

任何 mixed trade-off 默认保留为支线，不晋级。数值 margin 必须在 H3 读取 held-out
outcome 前冻结；本章程不按想象制造阈值。

## 8. 生产替换是另一项决策

研究主线晋级只改变论文/研究资源优先级。默认 App 替换仍需独立 Confirmation、
设备 parity、端侧稳定性、release gate 与风险说明。即使 HFTF H3 胜出，也不得直接接入
主动提醒、TTS、震动或安全主张。

H3 utility 胜出也不自动把 `INNOVATION_NOT_EVALUABLE` 改为创新性已确认；新颖性需要
独立系统检索、上述近邻直接比较和消融。

## 9. 停止与复用

- body/height 轴不超过 single-height：关闭该 multi-height teacher；
- future 轴不超过 current：关闭该 temporal formulation；
- history student 不能在同算力下保留 teacher/event 增量：关闭该 student；
- H3 未达 Pareto/non-inferiority：保持支线或关闭该 evidence version；
- 数据不具备绑定/真值：`NOT_EVALUABLE`，先修来源，不训练模型。

失败的 source audit、teacher atlas、student trace 和 challenger ledger 可作为
diagnostic/regression/negative evidence，但不得被重命名为 fresh Confirmation。
