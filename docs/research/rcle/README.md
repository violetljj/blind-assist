# RCLE 研究主线

状态：`current / ECOLOGICAL_DISCOVERY_ACTIVE`

最后核验：2026-07-28 19:20（Asia/Hong_Kong）

## 当前结论

RCLE-RF 仍是 BlindAssist 的论文研究主线，但研究方法已经从“理想数据合同驱动”
改为“数据能力驱动、分阶段提高证据强度”：

```text
CURRENT STUDY: RCLE_ECOLOGICAL_RESPONSE_DISCOVERY_R0
CURRENT TRACK: CAPABILITY_DISCOVERY
CURRENT RESULT ACCESS: OUTPUT_INSPECTED
CURRENT CLAIM CEILING: SINGLE_SESSION_DISCOVERY
SEALED EVALUATION: NOT_YET_ALLOCATED
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
| `CAPABILITY_DISCOVERY` | 观察真实响应、支持率和失败模式 | `ACTIVE` |
| `DEVELOPMENT_DIAGNOSTIC` | 在明确污染的数据上修实现、调候选 | `NOT_STARTED_FOR_R2` |
| `SEALED_EVALUATION` | 在算法与指标冻结后做 session 级独立评估 | `NOT_ALLOCATED` |

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

pair/frame 只是时间序列测量单位，不是独立统计样本。当前 600 pair 仍然只有一个
capture-session observation unit。未来比较必须按 session/route 聚合或分层，不能
用 pair 数膨胀样本量。

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
| 修改算法或阈值 | `NOT_YET_STARTED / MUST_RECLASSIFY_AS_TUNED_ON` |
| session 级 sealed evaluation | `NOT_ALLOCATED` |
| performance / generalization | `NOT_AUTHORIZED` |
| Android / host replay /主动告警 | `NOT_AUTHORIZED` |
| 真人、产品、安全或生产结论 | `NOT_AUTHORIZED` |

BlindAssist 仍是论文、毕业设计、院内演示和竞赛研究原型，不是可独立依赖的助行或
安全产品。

## 下一步

按信息增益顺序执行：

1. 在现有 ADVIO Discovery 上做旋转方向/坐标系、畸变和高低角速度的低成本机制审计；
2. 选择容易获得的自然行走 session，每个先跑 10–30 秒连续片段，不要求一个来源
   同时承担全部角色；
3. 根据真实失败模式决定是否进入 `DEVELOPMENT_DIAGNOSTIC`；
4. 在任何算法调试前，把未来 evaluation 按 person/session/route/sequence 原子单位
   写入能力表并标为 `SEALED_UNSEEN`；
5. 只有冻结算法、标签、指标、缺失处理和 session split 后，才运行
   `SEALED_EVALUATION`。

当前不继续旧公开数据市场漫游，不自动创建 formal claim，不进入 Android。
