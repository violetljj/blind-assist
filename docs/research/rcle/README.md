# RCLE 研究主线

状态：`current / TEMPORAL_STRUCTURE_R1_COMPLETE_HOLD / STANDALONE_ROTATION_ROUTE_STOP`

最后核验：2026-07-28（Asia/Hong_Kong）

## 当前结论

RCLE-RF 仍是 BlindAssist 的论文研究主线，但研究方法已经从“理想数据合同驱动”
改为“数据能力驱动、分阶段提高证据强度”：

```text
CURRENT STUDY: RCLE_TEMPORAL_STRUCTURE_DIAGNOSTIC_R1
CURRENT TRACK: DEVELOPMENT_DIAGNOSTIC
CURRENT RESULT: HOLD_MIXED_OR_INSUFFICIENT_TEMPORAL_EVIDENCE / VALID
CURRENT CLAIM CEILING: MULTI_SESSION_DESCRIPTIVE_TEMPORAL_STRUCTURE_AND_DEVELOPMENT_PRIORITY
AUDIT HISTORY: ROTATION_COMPENSATION_MECHANISM_AUDIT_R1_COMPLETE_NEGATIVE
AUDIT OUTCOME: STANDALONE_ROTATION_ROUTE_STOP_CONFIRMED_ACROSS_SESSIONS
SEALED EVALUATION: ADVIO_OFFICE04_SEQUENCE16_IPHONE_RESERVED_UNSEEN
ANDROID / PRODUCT / SAFETY: NOT_AUTHORIZED
```

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

时间结构 diagnostic R1 已完成，当前顺序为：

1. 正式结束 standalone rotation，不继续修补 rotation-only compensation；
2. 当前 fixed flow-quality gate 保持 `HOLD`，禁止按 response 调门直到通过；
3. 保留时间结构 `HOLD`：不把低 feature support 当统一解释，也不把 pose 周期代理
   当 gait 同步或因果证据；当前没有自动算法后继；
4. 不立即实现
   [reference-track failure diagnosis R0](RCLE_REFERENCE_TRACK_FAILURE_DIAGNOSIS_R0_DESIGN_2026-07-28.md)、
   temporal consistency 或 bearing；
5. sequence16 继续 `SEALED_UNSEEN`，不访问、不纳入本轮分析；
6. 不修改 `0.01/s`、三 pair 规则或 PairState，不进入路径走廊或 Android。

当前不继续旧公开数据市场漫游，不自动创建 formal claim，不进入 Android。
