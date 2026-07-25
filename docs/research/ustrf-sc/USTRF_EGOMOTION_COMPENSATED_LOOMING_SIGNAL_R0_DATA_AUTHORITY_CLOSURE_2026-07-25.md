# EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R0 数据权威收口（2026-07-25）

状态：`FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED / VALID`

最大权限：`R0_DOCUMENTATION_CLOSURE_ONLY`

## 决策

R0 按其冻结的“每个真实 source family 都必须覆盖四类 counterfactual cell、三种
role 与完整 session 分母”合同收口。ADT、AV2 与 CODa 的子边界分别得到：

- `ADT_CELL_PRESCREEN_INSUFFICIENT / VALID`：`0 / 5 / 0 / 0`；
- `AV2_REQUIRED_PURE_ROTATION_CELL_STRUCTURALLY_ABSENT / VALID`；
- `HOLD_CODA_BOUNDED_PRESCREEN / VALID`。

因此 R0 的新 source/session authority 无法闭合，只能使用其四个预注册合法终态
之一 `FAIL_CLOSED_NEW_DATA_OR_TRUTH_AUTHORITY_BLOCKED`。此前三来源组合名只保留为
非终态审计摘要，不是新的 R0 terminal。

## 结论边界

这是**数据合同与分母设计受阻**，不是 Looming 算法失败。R0 未运行 raw flow、
bbox growth、未补偿扩张、rotation-compensated expansion、oracle rotation 或
full-6DoF diagnostic；未冻结 role split，未选择阈值。

复核认为 R0 把信号存在性、跨来源复制和产品提醒三个阶段的要求叠加到了每个来源。
该合同读取 outcome 前已经冻结，所以不能在 R0 内事后改成分声明准入。修正必须另立
R1，保留本轮负的 source-availability 证据。

机器收据：
`artifacts.local/evidence/ustrf/egomotion_compensated_looming_r0/r0_data_authority_closure_terminal.json`，
SHA-256
`dd9943399f3b3b333ea63cf226e82283fd821fed32b74157a1b06fbbc67a37d9`。

## 不授权

- 不恢复 route-conditioned USTRF 或旧 15 对窗口；
- 不运行 App、route、event lifecycle、alert、shadow、人体或生产；
- 不继续 HOT3D/AV2/CODa 漫游式来源搜索；
- 不自动开始人体、视障参与者或自由行走采集。

## 后继

唯一后继是独立的
[claim-scoped Looming R1](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_CLAIM_SCOPED_EVIDENCE_GOAL_2026-07-25.md)。
R1 使用声明级准入、单元级 abstention、证据等级和跨来源证据拼图；它不继承 R0 的
全来源全 cell 分母，也不改写 R0 终态。
