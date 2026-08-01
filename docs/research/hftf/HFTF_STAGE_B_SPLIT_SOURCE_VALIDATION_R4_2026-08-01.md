# HFTF Stage B split-source validation R4

日期：2026-08-01

状态：`FROZEN_BEFORE_R4_OUTCOME`

## 1. 决定

R3.1 已证明当前 SANPO semantic-ground-only reference 在冻结预算内无法提供任何
step/drop opportunity，但 29/34 可计算 source 具有完整 obstacle opportunity。R4
不删除 ground、不降低门，也不继续扩大同一 outcome-open 队列；它把两个不同的
source role 显式拆开：

- SANPO-Synthetic fresh sessions：只评价真实渲染场景中的 foot/body/head
  swept-envelope obstacle effect；
- deterministic analytic metric terrain：只评价 foot-ground continuity 对
  rise/drop、局部 bump/pit、平地/坡道/阈值内反例和 UNKNOWN 的 mechanics。

两个门同时通过，才允许终态
`R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED`。它只允许冻结 Stage C source
feasibility contract，不直接授权执行 Stage C 或训练 student。

## 2. Obstacle source role

排除 R0–R3.1 全部 56 个 outcome-open sessions 后，继续从 official
SANPO-Synthetic train 按完整 session ID 字典序选择 inventory-eligible source。最多
screen 12 个、目标 4 个；取得第 4 个即停止。

qualification 只能读取 disjoint stride-4 dense obstacle reference。每个 height 必须
known coverage `>=.10`、primary positive `>=5`、negative `>=20`，并在 reference
threshold `1/2/4/8` 上同时有 micro positive/negative。selection 不得读取 ground、
stride-8 candidate、angular baseline 或任何 arm delta。

正式 comparison 保持 R3 的同点 baseline、公平 reference 与全部 effect gates：
cohort F1/precision delta `>=+.10`、recall delta `>=-.02`、4/4 session F1 delta
`>=+.05`，每 height 和四个 sensitivity directions 全部一致。

## 3. Terrain source role

解析真值在 observation sampling 前由冻结的五段高度 profile 定义；semantic class
不参与真值。42 个 cases 精确包含：

- 16 个 no-risk：flat、traversable ramp、阈值内 rise/drop；
- 20 个 risk：hazardous rise/drop、endpoint 净变化为零的 localized bump/pit；
- 6 个只支持 3/5 sections 的 occluded UNKNOWN。

candidate 复用 D0 的 five-section median、`rise>.18 m`、`drop<-.15 m`、至少 4 个
supported sections 和 UNKNOWN→SAFE firewall。基线为：

1. `semantic_support_is_safe`；
2. 只比较首尾高度的 `endpoint_elevation_delta`。

candidate 必须 precision/recall/F1/specificity 均 `>=.95`、每个危险 family recall
`>=.95`、UNKNOWN abstention `=1.0`、对最佳 baseline 的 F1 delta `>=+.15`，且没有
UNKNOWN→SAFE。

这只是 controlled mechanics evidence。通过不能声称自然场景台阶 prevalence、真实
深度噪声鲁棒性、助盲事件效用或 safety。

## 4. 停止与权限

顺序终态：

1. `R4_OBSTACLE_OPPORTUNITY_COHORT_NOT_EVALUABLE`
2. `R4_OBSTACLE_ENVELOPE_GAIN_NOT_SUPPORTED_STOP`
3. `R4_ANALYTIC_TERRAIN_MECHANICS_NOT_SUPPORTED_STOP`
4. `R4_STAGE_B_SPLIT_SOURCE_TEACHER_MECHANICS_SUPPORTED`

在 joint success 前不授权 Stage C。joint success 后只允许冻结 source-feasibility
contract；student training、研究主线、Android、提醒、默认 App、生产和安全 claim
均仍为 false。
