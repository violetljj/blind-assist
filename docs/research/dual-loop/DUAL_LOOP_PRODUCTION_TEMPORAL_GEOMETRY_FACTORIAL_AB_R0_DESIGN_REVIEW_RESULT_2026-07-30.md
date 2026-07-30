# Production temporal geometry factorial A/B R0 设计复核结果

状态：`PASS / CONTRACT_FROZEN / IMPLEMENTATION_AUTHORIZED / NOT_RUN`

日期：2026-07-30（Asia/Hong_Kong）

## 结论

两项相互独立的只读复核均在修订后给出 `PASS`。冻结合同为
[DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_PROTOCOL_2026-07-30.json](DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0_PROTOCOL_2026-07-30.json)，
SHA-256：
`897329d334a7bfd4654358a7e5948e0ae8cebd3614ec110d61d073e0ccfa30ae`。

该 PASS 只允许实现 factorization、合成 mutation tests、implementation lock 与后续
activation review；不允许正式 decision RGB 执行、truth join、Confirmation、生产行为
变更或产品/安全主张。

## 为什么改走这条路线

旧 Sparse LK F-1B 已以 `NO_INCREMENT / VALID` 关闭，LITE R2 又在 469 个 parent
event 上得到两臂均不达 readiness floor 的有效负结果。继续把失败的 radial-flow
候选接入 Android 只会增加实现距离。

现有生产 `TemporalRiskTracker` 已经在同一 detector target 上使用最多 5 帧 /
900 ms 的框底部、框面积与可选深度趋势，并进入现有稳定、反馈与冷却链。R0 因而只做
一个可归因的生产因子实验：

- A：更新同样 tracker history，但只中和 `DetectionSource.OBJECT_DETECTOR` 的
  temporal output；
- B：保持当前完整生产 tracker；
- 每帧 QNN 只推理一次，同一 immutable detections 同时供给两个全状态隔离分支。

第二环必须称为 `detector-conditioned temporal geometry`，不是独立传感器。

## 修订后通过的关键门

- 两个 CrowdBot capture 来自一个 collection context，不宣称两个总体独立复制；
- outcome-blind RGB receipt 验证 `4422/4422` PNG、逐文件哈希、严格时序、
  `640×480` 和空 candidate namespace；
- truth-membership receipt 冻结 17 项原始 truth，预先排除无有效帧的
  `F1A-P-007/P-009`，正式评分分母为 8 positive + 7 negative；
- EARLY_RESPONSE 与 RISK_DISCRIMINATION estimand、整数纳秒边界、可评价条件、
  no-harm guardrail、联合 terminal 与部分不可评价 terminal 均可唯一复算；
- A/B kernel、tracker、stabilizer、event、confirmation、FeedbackController、
  fatigue 与 trace 全隔离；合成测试还必须证明 segmentation parity 和执行顺序不变性；
- 正式 producer 不接受 truth 输入；只有独立 validator 封存 `8844` 行 truth-blind
  trace 后，独立 evaluator 才可读取 truth；
- QNN HTP FP16、2.47.0、生产 model token 与 no-fallback 规则被冻结；
- formal-start 前失败为 `NOT_EVALUABLE_PRESTART`，之后任何失败为
  `INVALID_EXECUTION`，同一 evidence version 不得重跑。

## 冻结输入收据

| receipt | status | SHA-256 |
| --- | --- | --- |
| RGB input identity | `VALID / 4422` | `32c80d61bdedf0fa678d09a25e43d84232c4976fd5af0a644bb579c350d4d910` |
| truth membership | `VALID / 17 → 15 scored` | `42f36add7863a16210b4c0add41060ede94a50787591f1744bdb9a8aabce5290` |

两份 receipt 均记录 candidate output 未打开，四个新候选 output namespace 均不存在。

## 当前边界

```text
DESIGN_REVIEW: PASS
CONTRACT: FROZEN
IMPLEMENTATION: AUTHORIZED_NOT_IMPLEMENTED
IMPLEMENTATION_REVIEW: NOT_RUN
FORMAL_EXECUTION: NOT_AUTHORIZED
CANDIDATE_OUTPUT: NOT_ACCESSED
CLAIM_CEILING: TWO_SESSION_ONE_CONTEXT_DEVELOPMENT_DIRECTIONAL_SCREEN_ONLY
```
