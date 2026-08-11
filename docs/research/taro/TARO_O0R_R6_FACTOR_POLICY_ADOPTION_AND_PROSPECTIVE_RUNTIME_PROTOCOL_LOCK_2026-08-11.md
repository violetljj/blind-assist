# TARO O0R R6 factor-policy adoption and prospective-runtime protocol lock

状态：`PROTOCOL_FROZEN / IMPLEMENTATION_ALLOWED / EXECUTION_NOT_AUTHORIZED`

机器合同：[JSON](TARO_O0R_R6_FACTOR_POLICY_ADOPTION_AND_PROSPECTIVE_RUNTIME_PROTOCOL_LOCK_2026-08-11.json)

## 为什么 R6 PASS 后还需要这一层

R6 已在 8 个 untouched parents 上确认 factor ownership：SUPPORT/BOUNDARY 使用 Phase-A selected component，
QUERY_CLEARANCE 固定使用 R1 baseline。但 R5/R6 task-metric extractor 仍在 FARO 定义的 common-support pixel IDs
上比较 candidate；这适合受控归因，不是 source-only runtime interface。

本协议关闭这一条剩余接口缝：prospective factor runtime 的 point IDs、local surface、coverage 和 validity
全部从 factor owner 自己的 sealed depth、K、重力与 query receipt 重算。public API 不接受 FARO、truth、task
metric 或 prior outcome。

## 冻结 source-defined extractor

- baseline 使用 sealed raw DepthART metric depth；direct boundary 使用 source-scale anchored depth；
- baseline support 在 raw candidate 的 source geometry 上拟合；direct support exact-copy Phase-A Apple plane；
- 每个 owner depth 每帧只 unproject 一次，九个 query 共享 frame geometry；
- local surface、support、boundary、query clearance 的范围、stride、点数、斜率、coverage 与 capsule 常数全部
  机器冻结在 JSON；
- factor block 必须分别绑定实际 depth SHA、source-surface pixel-ID SHA 和 query receipt；
- 任一 validity gate 失败都保留 `UNKNOWN`，不删除样本、不计作 negative；
- 本层不接 uncertainty model、不运行三态 reducer、不允许输出 `CLEAR`。

## 防止二次使用 untouched outcome

刚完成 R6 confirmation 的 8 个 parent 已显式列为 implementation、formation、调参和 threshold selection 禁用。
implementation 只允许 synthetic fixtures；未来若做 formation replay，只能使用既有 24 个 formation parents，且
`promotion_allowed=false`。任何新的 formal confirmation 必须另选至少 8 个 parent-disjoint parents，并另签 data-use、
source hash、model/truth execution lock。

## 唯一后继

`TARO_O0R_R6_PROSPECTIVE_FACTOR_RUNTIME_IMPLEMENTATION_LOCK`

下一步只实现 source-defined factor runtime、validator 与 synthetic mutation tests。不得下载新数据、运行模型、
打开 R6 untouched outcome 作为开发输入、执行 formation replay 或 truth scoring。即使实现通过，claim ceiling 仍只到
source-only interface mechanics，不包含 calibrated uncertainty、final clearance、部署、产品或安全。
