# RCLE 研究主线

状态：`current / STAGE_B_COMPLETE_INDEPENDENT_VALID / B_ORACLE_NOT_EVALUABLE / C_D_CLOSED / SUCCESSOR_FORMAL_NOT_CONSUMED`

最后核验：2026-07-29（Asia/Hong_Kong）

## 当前结论

RCLE-RF 仍是 BlindAssist 的论文研究主线，但研究方法已经从“理想数据合同驱动”
改为“数据能力驱动、分阶段提高证据强度”：

```text
CURRENT STUDY: RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2
CURRENT TRACK: CONTROLLED_COUNTERFACTUAL_DEVELOPMENT
SCIENTIFIC STATUS: QUALITY_CALIBRATION_PASS
PROTOCOL STATUS: VALID
EXECUTION AUTHORITY: QMS_R1_SUCCESSOR_FORMAL_AUTHORIZED_ONE_SHOT / NOT_RUN
DEV DIAGNOSTIC: PERIODIC_SELF_MOTION_SENSITIVITY_OBSERVED / 8_OF_8_CLUSTERS
MOTION COMPONENT STAGE A: COMPLETE / VALID
STAGE B: COMPLETE / INDEPENDENT VALID / B_ORACLE_NOT_EVALUABLE
STAGE B ROTATION BOUNDARY: 0_OF_8_PASS
STAGE B REQUIRED COVERAGE FAILURES: 18_ARM_CLUSTERS
FEATURE CONTRACT C / FUSION D: CLOSED
CURRENT CLAIM CEILING: CONTROLLED_GENERATOR_INTERNAL_MECHANISM_DEVELOPMENT_ONLY
PREDECESSOR RESULT: TEMPORAL_STRUCTURE_R1_HOLD_MIXED_OR_INSUFFICIENT / VALID
AUDIT HISTORY: ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_COMPLETE_NEGATIVE
AUDIT OUTCOME: STANDALONE_ROTATION_ROUTE_STOP_CONFIRMED_ACROSS_SESSIONS
SEALED EVALUATION: ADVIO_OFFICE04_SEQUENCE16_IPHONE_RESERVED_UNSEEN
FORMAL R2 EXECUTION: NOT_AUTHORIZED
ANDROID / PRODUCT / SAFETY: NOT_AUTHORIZED
```

旧 R2 P4 已以
`INTERVENTION_NOT_EVALUABLE / VALID / COMPLETE_PRE_R3_TERMINAL` 永久关闭，
且正式 R3 pair-core call 为零；这不是 RCLE 算法 outcome。后继 QMS-R1 只修复
quality manipulation estimand：固定材质内部 residual contraction，在旧
development identities 上 `160/160`、全新 disjoint CAL 上 `32/32` 通过，
八个 subgroup 均为 `4/4`，512 个 frame state 的 prequantization relation
误差均为零。独立 validator 与 11 个 mutation tests 通过。随后 activation
preflight 冻结了全域不相交的新 480+16 和固定 8 条 PREFLIGHT identities，
复验 all-seed geometry、R3 transport 与 analysis lock 均未漂移；W8 完整实测
`1099.9671 s`，按 QMS-R1 shared-render scheduler 含 10% reserve 的保守投影为
`11.3375 h`。独立终态为
`ACTIVATION_PREFLIGHT_PASS / VALID / FORMAL_NOT_RUN`，已签发一次性 successor
formal authority，但正式 496 条仍为零。详见
[activation preflight R0](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_FORMAL_ACTIVATION_PREFLIGHT_R0_RESULT_2026-07-29.md)
与
[QMS-R1 结果](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_MANIPULATION_SUCCESSOR_R1_RESULT_2026-07-29.md)。

后续默认算法研究顺序已另立为
[RCLE 新算法研究推进路线 R0](RCLE_ALGORITHM_RESEARCH_ROADMAP_R0_2026-07-29.md)：
先做 clean 四臂运动分量定位，由 A 决定 B 的诊断重点；B 后必须在
`go single-upgrade / freeze residual feature / stop-or-downgrade / not-evaluable`
之间形成正式决定。只有前序证据购买复杂度时，才允许一个 base-vs-upgrade 候选；
C 只冻结最终保留版本，D 仅在独立标签和 session/route 划分成立时进入。新路线不
改写旧结果或既有 QMS 权限。A 的两批各完成 `4 clusters / 16 sequences /
9,616 pairs`，独立 validator 从 cell primitives 重建 pair 指标：
rotation absolute leakage 与 translation signed response 均为
`4/4 → 4/4` 正方向，full interaction 为 `2/4 → 3/4` 且跨批不稳定。第三个
独立 closeout validator 已给出 `VALID / STAGE_A_COMPLETE`。随后 Stage B
translation-depth oracle + object-approach control 完成了 contract preflight、
geometry materialization、独立几何门、一次性 activation 和完整执行：
`8 clusters / 40 sequences / 24,040 pairs`。独立 validator 从 sealed tracks
重算 `24,040` 个 pair 与 `865,440` 个 cell fit。translation oracle 在
rotation-only 臂保持严格 no-op（`u_T=0`、baseline/oracle 完全相同），但 unchanged
R3 rotation absolute P90 为 `0.0940–0.1806/s`，冻结的 `<=0.01/s` 必过边界
`0/8` 通过；另有 `18` 个 required arm-cluster coverage 低于 `0.75`。因此终态为
`B_ORACLE_NOT_EVALUABLE`，single upgrade、C、D 与 retry 均关闭。详见
[Stage B result R1](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_RESULT_R1_2026-07-29.md)
与历史
[Stage B contract preflight R0](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_STAGE_B_TRANSLATION_DEPTH_ORACLE_OBJECT_APPROACH_CONTRACT_PREFLIGHT_R0_RESULT_2026-07-29.md)。
`480+16` 仍 `NOT_CONSUMED / NOT_RUN`。详见
[motion-component Stage A](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_CLEAN_MOTION_COMPONENT_LOCALIZATION_R0_STAGE_A_RESULT_2026-07-29.md)。

为避免直接支付约 11 小时正式预算，随后冻结并运行了与所有旧 formal、DEV、
CAL、PREFLIGHT 及 successor formal identities 全域不相交的四-block DEV
diagnostic：`8 clusters / 48 sequences / 28,848 pairs`，完整保留每条
`601-pair` 时间结构。W8 在约 `37.8 min` 完成，独立 validator 终态为
`VALID / DEV_DIAGNOSTIC_COMPLETE`。`MOTION_CLEAN` trigger-density contrast
在 8/8 clusters 为正，均值 `0.25`、范围 `0.18–0.29`；质量 interactions
方向混合。该结果只说明受控 generator 内部的周期自运动敏感性，不是正式推断。
successor formal 一次性授权仍未消费，正式 480+16 仍为零。详见
[four-block DEV diagnostic R0](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QMS_R1_FOUR_BLOCK_DEV_DIAGNOSTIC_R0_RESULT_2026-07-29.md)。

轻量 P3 R0 已完成 R3 transport equivalence、analysis implementation/mutation
tests，以及固定 8 个 PREFLIGHT identities 的 guarded-host qualification。
初始 `8 -> 496` 均匀外推错误放大 guardrail 比例，已由不可覆盖的 scheduler
successor 修正。优化后 W8 完整实测 `677.5074 s`，全量 transport hashes 与
predecessor 精确一致；按 `480 factorial + 16 guardrail` 分项外推并计入 10%
retry reserve 后为 `7.1575 h`。独立终态为
`PERFORMANCE_QUALIFIED / VALID / P4_NOT_ACTIVATED`，选择 W8。
详见
[P3 R0 结果](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_TRANSPORT_ANALYSIS_RUNTIME_PREFLIGHT_R0_RESULT_2026-07-29.md)。

P2 response-blind quality calibration 的一次性 blur-grid repair R1 已完成。
复用原 CAL panel，仅运行冻结的 9 个小 sigma 候选，共 `5120` 行 blur ledger；
`sigma=0.475 px` 是满足总体与全部 8 个 block×motion subgroup 门的最小候选，
并与 hash-bound R0 `alpha=0.15` 形成全局 strength pair。独立复算
`errors=[]`，终态为
`QUALITY_CALIBRATION_PASS / VALID / P3_NOT_AUTHORIZED`。未运行 RCLE 或 P3，
也未重调 low-texture、换 seed、分 block 或自动二次修复。详见
[P2 R1 结果](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_RESULT_2026-07-29.md)；
[P2 R0](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_R0_RESULT_2026-07-29.md)
保留为不可变 predecessor。

P1 最新冻结回执为隔离的 `R2_KEYSET_REPAIR_R0`
`95646437fbe0ef0cf03844f94467303f5d90ca15c3e22fc1785157b037a8c079`：
G01–G14 为 14/14 PASS，`errors=[]`，终态为
`GENERATOR_GEOMETRY_PASS / EXECUTION_NOT_AUTHORIZED`。本版只把历史 R0
evidence key 修正为真实的 `producer_receipt.json`，并加入 generator directory
与正式 receipt 的独占创建保护；88 条 all-seed record 与 R2 逐字节一致。
R0、R1、R2 失败回执仍不可覆盖；P1 本身不曾自动授权 P2。详见
[keyset-repair 结果](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_KEYSET_REPAIR_R0_RESULT_2026-07-29.md)
与历史 [R2 结果](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R2_RESULT_2026-07-29.md)。
R1 的不可变 13/14 失败与并发 source-hash 竞态见
[R1 结果](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GEOMETRY_SPEC_REPAIR_R1_RESULT_2026-07-28.md)。

当前操作只看三轴：科学状态仍为 `QUALITY_CALIBRATION_PASS`、successor
activation preflight 为 `VALID / FORMAL_NOT_RUN`、执行权限为
`QMS_R1_SUCCESSOR_FORMAL_AUTHORIZED_ONE_SHOT`。历史 predecessor P3 的
`P4_NOT_ACTIVATED` 与旧 P4 的 consumed terminal 继续保留。历史 R2 的
`INVALID_KEYSET` 不再被表述为几何失败。P1
已经关闭；非阻断的 receipt、命名或未来漂移监控只进入 backlog，不再创建 P1
版本或阻挡算法阶段。

旧公开数据确认合同仍永久保持：

`RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE / VALID_FAIL_CLOSED_TERMINAL`

这两个事实同时成立：

1. 旧 R1 的两个冻结片段没有形成 eligible RGB，不能证明算法成功或失败；
2. 用户已另行授权新的 Ecological Discovery，并在自然连续视频上实际运行算法。

新实验不是旧 R1 的重试、救援或回写。

## 已采用的研究方法 R2

权威方法见：

- [数据能力驱动 RCLE 主线 R2](RCLE_DATA_DRIVEN_RESEARCH_MAINLINE_R2_2026-07-28.md)；
- [全局研究治理](../../RESEARCH_GOVERNANCE.md)；
- [当前轻量能力表](RCLE_ACTIVE_DATA_CAPABILITY_MAP_R1_2026-07-28.csv)。

### 三条数据轨道

| 轨道 | 用途 | 当前状态 |
| --- | --- | --- |
| `CAPABILITY_DISCOVERY` | 观察真实响应、支持率和失败模式 | `NATURAL SESSION R0 COMPLETE` |
| `DEVELOPMENT_DIAGNOSTIC` | 在明确污染的数据上修实现、调候选 | `ROTATION R1 COMPLETE / REFERENCE TRACK DEFERRED` |
| `SEALED_EVALUATION` | 在算法与指标冻结后做 session 级独立评估 | `ADVIO sequence 16 RESERVED / NOT EXECUTABLE` |

跨来源测试单独称为 `EXTERNAL_TRANSFER`，不再和普通同来源 session holdout 混为
一谈，也不是当前 Discovery 的前置条件。

### 四级结果访问

| 状态 | 允许用途 |
| --- | --- |
| `CONTENT_INSPECTED` | 看内容但未看目标算法输出；可在预冻结后进入 evaluation，并披露筛选依据 |
| `OUTPUT_INSPECTED` | 已看 RCLE 或 baseline 输出；Discovery / Development |
| `TUNED_ON` | 已用于改算法、阈值、窗口或指标；Development only |
| `SEALED_UNSEEN` | 未看算法输出，且算法和指标已冻结；Evaluation |

同一来源的新 person、capture session、route 或 sequence 可以构成独立 holdout。
连续帧随机切分或把同一长视频切成多个 clip 不能构成独立样本。

pair/frame 只是时间序列测量单位，不是独立统计样本。Natural-session R0 的
`4 × 601` pair 仍然只有四个 capture-session observation units。未来比较必须按
session/route 聚合或分层，不能用 pair 数膨胀样本量。

## 当前 Discovery

`RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0` 的宽松观察清单是：

```text
观察接近、正常行走、转头、横穿、模糊、低纹理和步态振荡下，
bbox growth、raw expansion 与 RCLE 的响应分布、支持率、
触发密度、时间一致性和失败案例。
```

Discovery 允许 RCLE 胜出、部分有效、没有优势，或者被更简单 baseline 超过。当前
不预设分类阈值、AUROC/F1 目标或算法晋级。

第一轮已在 ADVIO office03 sequence 15 的预声明起始连续 `9.999266 s` 上运行：

- 原生约 60 Hz，600 个连续 pair，599/600 可评估；
- raw 与 rotation-compensated 三连续触发比例均为 `0.4000`；
- absolute response 对角速度的 Spearman 为 raw `0.3498`、compensated `0.3804`；
- 未观察到 rotation compensation 在该片段中降低触发或角速度关联；
- `bbox_growth` 因没有冻结目标框为 `NOT_EVALUABLE`。

详细限制和结果见
[首轮 Discovery 结果](../../../scripts/research/egomotion_compensated_looming/ecological_response_discovery_r0/RESULT_2026-07-28.md)。
它是一个已查看输出的单 session、半分辨率、分块执行诊断，不能产生 performance、
generalization 或 causal confirmation。

机制审计已完成，详见
[R1 结果](../../../scripts/research/egomotion_compensated_looming/rotation_compensation_mechanism_audit_r1/RESULT_2026-07-28.md)：

- 首轮把官方 `wxyz` 当成 `xyzw`，并遗漏 `T_cam_imu` pose-to-optical basis；
- `R_current.T @ R_previous` 与 current-to-previous warp 的正负号本身正确；
- 合成 yaw/pitch/roll 的 correct arm 全部优于 raw/reverse；
- 最终 R3 在原始/去畸变高角速度窗把三-pair 触发分别从
  `0.7083→0.9417`、`0.7083→0.8417`；
- 去畸变影响总体响应，但不救回补偿；七 chunk 状态重置已由单进程连续运行消除。

因此当前独立 rotation-compensation 路线停止，允许形成论文级负结果。受控旋转有效
只支持把 RCLE 保留为局部机制特征，不能证明其单独足够。未访问的 ADVIO office04
sequence 16 已在修实现前原子预留为 future `SEALED_UNSEEN`，在算法和指标冻结前
禁止下载、解码或运行。

随后完成
[natural-session expansion Discovery R0](RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0_RESULT_2026-07-28.md)：

- metadata-only 固定 ADVIO sequence13、14、15、17 为 Discovery/Development，
  sequence16 保持 `SEALED_UNSEEN`；
- 每个 session 只运行一个 `10.0159–10.0175 s`、601-pair 连续片段，不分块、不换片；
- strict `> 0.01/s`、三连续 pair、单一连续 `PairState` 与 R3 几何实现均未改；
- support 为 `0.9867–0.9967`，各 session 只分别报告响应、固定分母触发密度、角速度
  关联和 common-grid support 失败；
- 在各 session 最高 20% 角速度层中，sequence13、15、17 同时出现 compensated
  触发密度和 absolute response 高于 raw，达到预冻结 `>=2 sessions` 停止规则；
- sequence14 未恶化；静态接近、横穿和模糊因无冻结事件标签保持
  `NOT_EVALUABLE`，没有事后换 clip。

因此 standalone rotation 路线已由多个自然 session 正式停止。reference-track
设计保留为历史 design-only 资产，但不再是当前顺序，也未获得实现权限。

下一步的
[退化归因与 flow-quality diagnostic R0](RCLE_DEGRADATION_FLOW_QUALITY_DIAGNOSTIC_R0_RESULT_2026-07-28.md)
也已完成。它在相同 pair 身份上保持 R3、`>0.01/s` 和三 pair 不变，并先从 RGB/pose
生成不读取 response/风险标签的 blur、texture、gait 与 flow-quality 代理。高响应
最一致地集中在 gait proxy（`3/4` session），blur 和 low texture 各为 `2/4`；
fixed flow gate 只有 `1/4` session 富集高响应，所有 session 的 trigger-density
下降均小于预冻结 20% 门，终态为 `HOLD_FLOW_QUALITY_GATE / VALID`。这不恢复
rotation-only 路线，也不把 gate 拒绝称为假警。

随后完成
[时间结构诊断 R1](RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1_RESULT_2026-07-28.md)。
它在正式输出前冻结 `0.7–3.0 Hz` signed pose、全局/径向 flow 方向、周期、轴向相位
锁定和 collapse event，并保持同一四 session、同一 pair 身份与两阶段防火墙。
四 session 的 pose band-energy fraction 为 `0.729–0.924`，flow direction 覆盖
`75.4%–99.2%`、相邻方向余弦 `0.976–0.993`；但 flow-at-pose-frequency
`R²` 只有 `0.020–0.035`，高响应与 measurement-failure overlap 只有
`17.6%–47.1%`。motion routing 与 quality routing 均为 `0/4`，终态
`HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE / VALID`。这既不支持把高响应主要
归于 collapse，也没有证明与 pose-derived 周期同步。

当前后继已冻结并通过两路隔离设计审查：
[周期性自运动反事实 R2](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_PREREGISTRATION_2026-07-28.md)
的核心设计：四个 ADVIO pose 波形只作为 response-blind motion block，20 个
block-specific 新 3D scene seed 在
`static/periodic 6DoF × clean/blur/low-texture` 六臂内严格配对。480 条序列只是
运行规模，统计单位为 80 个 `scene_seed × motion_block` cluster；五个主要对比使用
block-stratified paired bootstrap 与 familywise simultaneous interval。机器
[合同](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json)、
[3D geometry spec](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GEOMETRY_VALIDATION_R0_2026-07-28.json)
和
[run budget](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_RUN_BUDGET_R0_2026-07-28.json)
及
[design review result](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_DESIGN_REVIEW_RESULT_2026-07-28.md)
均保持 `formal_execution_authorized=false`。随后另立的
[generator geometry P1 R0](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R0_RESULT_2026-07-28.md)
已实现 deterministic non-planar 3D generator、fixture、80+8 all-seed manifest
与独立 validator。G01–G12、G14 通过；G13 因 frozen exact 25% inverse-depth
endpoint 只对应约 `0.0223/s` radial expansion（低于 `0.05/s`），且
approach-plus-periodic 不保持 pairwise monotonic depth 而失败。终态为
`INTERVENTION_NOT_EVALUABLE / HOLD_P1`。后续 R1/R2 失败身份保持不可变；隔离的
[P1 keyset repair](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_KEYSET_REPAIR_R0_RESULT_2026-07-29.md)
已在不改 88 条 scene record、seed、trajectory、gate 或阈值的前提下完成 14/14
独立验证，终态为 `GENERATOR_GEOMETRY_PASS / EXECUTION_NOT_AUTHORIZED`。
quality calibration 已按冻结 grid 合法关闭；synthetic transport、analysis
implementation、runtime preflight 和正式序列仍未创建或运行。

## Discovery 的低成本操作门

数据进入 Discovery 只要求：

- 可以取得并解码；
- 时间顺序可复算；
- dataset/sequence 身份基本明确；
- 已知许可或使用限制有记录；
- 下载和适配成本有界。

固定十秒、同源正负、精确物理闭合率、同时具备 RGB/pose/depth，以及一个来源覆盖
全部角色，都不再是默认准入条件。缺少某模态只降低可回答问题和 claim ceiling。

能力表只有 10 列 CSV，不是运行许可证，不再开发通用数据管理框架或复杂 adapter
体系。

## 旧终态与历史资产

[历史数据工作收束报告 R0](RCLE_DATA_WORK_CLOSURE_R0_2026-07-28.md)和
[历史 19 列能力库存 R0](RCLE_DATA_CAPABILITY_MAP_R0_2026-07-28.csv)继续作为 archive。
它们不再决定新 Discovery 的数据准入，但所有旧 terminal、claim、失败 receipt 和
访问事实保持不可变。

旧 R1 仍具有以下事实：

- eligible RGB frame = 0；
- pixel decode = 0；
- RGB algorithm call = 0；
- alignment denominator = 0；
- 两个旧 claim 已消费，禁止重试、换窗、扩预算或整源回退。

## 当前权限

| 能力 | authority |
| --- | --- |
| 自然视频 Capability Discovery | `AUTHORIZED / ACTIVE` |
| 已查看数据的失败分析和回归 | `AUTHORIZED` |
| natural-session expansion Discovery R0 | `COMPLETE / VALID` |
| degradation / flow-quality diagnostic R0 | `COMPLETE / HOLD_FLOW_QUALITY_GATE / VALID` |
| temporal-structure diagnostic R1 | `COMPLETE / HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE / VALID` |
| periodic self-motion counterfactual R2 predecessor | `P4: INTERVENTION_NOT_EVALUABLE / VALID / COMPLETE_PRE_R3_TERMINAL` |
| QMS-R1 successor formal | `AUTHORIZED / ONE_SHOT / NOT_RUN / OPERATOR_HOLD_NOT_CONSUMED` |
| 新算法关卡路线 | `ADOPTED / GATED_PROCESS / STAGE_A_COMPLETE` |
| A Stage 1 / Stage 2 | `VALID_COMPLETE / VALID_COMPLETE` |
| B translation-depth oracle | `COMPLETE / INDEPENDENT_VALID / B_ORACLE_NOT_EVALUABLE` |
| B 后升级决策 | `CLOSED / ROTATION_BOUNDARY_0_OF_8 / COVERAGE_FAILURES_18` |
| 单项升级 / C / D | `NOT_AUTHORIZED / CLOSED / CLOSED` |
| rotation compensation R3 机制审计 | `COMPLETE / STANDALONE ROUTE STOP CONFIRMED ACROSS SESSIONS` |
| reference-track failure diagnosis R0 | `DEFERRED / DESIGN_ONLY / NOT_AUTHORIZED_TO_EXECUTE` |
| 修改 `0.01/s` 或三 pair 规则 | `NOT_AUTHORIZED` |
| session 级 sealed evaluation | `RESERVED_NOT_EXECUTABLE` |
| performance / generalization | `NOT_AUTHORIZED` |
| Android / host replay /主动告警 | `NOT_AUTHORIZED` |
| 真人、产品、安全或生产结论 | `NOT_AUTHORIZED` |

BlindAssist 仍是论文、毕业设计、院内演示和竞赛研究原型，不是可独立依赖的助行或
安全产品。

## 下一步

当前默认顺序为：

1. A 已完成并独立收口；Stage B 也已完整执行和独立重建；
2. Stage B 的 must-pass rotation boundary 为 `0/8`，required coverage failures
   为 `18`，终态固定为 `B_ORACLE_NOT_EVALUABLE`；
3. 不重跑、不替换 identity、不改 R3/阈值/三 pair/abstention，也不把边界失败后的
   translation 或 positive-control 描述量用于购买升级；
4. single targeted upgrade、feature contract C 与 fusion experiment D 均关闭；
5. 当前只做论文/机制审计负结果收口；任何新科学问题必须另立授权与合同，不能沿用
   本次 response 做事后救援；
6. successor formal `480+16` 仍为 `NOT_CONSUMED / NOT_RUN`，sequence16、Android
   和产品权限继续关闭。

当前不继续旧公开数据市场漫游，不自动创建 formal claim，不进入 Android。
