# P1-W2 referent anchor interface feasibility protocol V1

状态：`DESIGN_FROZEN / IMPLEMENTATION_NOT_SELECTED / DATA_NOT_SELECTED / NO_EXECUTION / NO_STAGE_A_V2 / NO_MEMORY_WRITE / DEFAULT_APP_UNCHANGED`

Claim ceiling：`ANCHOR_INTERFACE_DESIGN_ONLY / NO_EMPIRICAL_CAPABILITY / NO_PERSISTENCE_OR_WORLD_MEMORY / NO_PRODUCT_OR_SAFETY_AUTHORITY`

## 1. 唯一问题

P1-W2 只回答：

> 在一次合法 P0 referent handoff 之后，单个现实目标能否形成足够的 anchor evidence，使后续视图既保留 referent-local spatial support，又能与 same-scene identity confuser 分开；证据不足时能否诚实 abstain？

本阶段研究的是 `referent anchor formation`，不是 tracker、reacquisition、keyframe memory 或 world-relative persistence。
P1-W1 Stage A v1 已消费；其 `W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE` 不因本协议而改写。

## 2. 固定输入与观察单位

- source 是一个通过既有 P0 contract 的 handoff，绑定 candidate、source frame、referent region 与 evidence lineage；
  evaluator-only exact identity、future frame、后续 truth region 均不可进入 provider。
- source 只允许一个冻结 observation。多帧初始化、滚动模板、online update 与 keyframe bank 不属于 P1-W2。
- probe 是与 source 独立配对的后续真实视图。rotation、translation、reappearance 与 same-scene confuser 分桶，但不按
  时间序列运行 policy，也不继承前一 probe 的状态。
- 基本分析单位是 `source referent × probe view` pair；同一 source 的多个 probe 是相关重复，不能当作独立样本扩张
  denominator。

## 3. 两条证据路径

### 3.1 Anchor observability

任何 detector-free、semi-dense 或 dense matcher 都只是 correspondence candidate provider：

```text
pairwise matcher
  -> correspondence candidates
  -> source/probe region filtering
  -> referent-core support
  -> spatial dispersion
  -> geometric-model consistency
  -> GEOMETRY_SUPPORTED | GEOMETRY_UNSUPPORTED | NOT_OBSERVABLE
```

match count、matcher confidence 或可视化上“看起来很多匹配”均不具有 geometry authority。正式 implementation/data
selection 必须在看 outcome 前冻结 exact checkpoint、输入分辨率、candidate filtering、最小 core support、dispersion、
geometric model、residual/inlier gate、degeneracy 与 abstention 规则。

重复门窗上的高数量错误匹配必须由 referent-local support 与 geometric consistency fail closed；不得以 context-only
一致性替代 referent-local geometry。

### 3.2 Identity separability

任何 DINO-style dense feature、segment representation 或 region embedding 都只是 identity evidence provider：

```text
referent core representation
  -> true cross-view score
  -> hardest same-scene confuser score
  -> frozen separation/abstention rule
  -> IDENTITY_SEPARATED | IDENTITY_AMBIGUOUS | NOT_OBSERVABLE
```

通用 image/pixel benchmark、embedding similarity 或模型名称均不授予 physical identity authority。正式选择必须冻结
checkpoint/license/provenance、core construction、pooling、normalization、score、hardest-confuser definition 与 separation
rule；truth identity 只能由 evaluator 在 provider 输出完成后读取。

## 4. Core 与 context 的权限

```text
referent core / admitted mask
  identity evidence: allowed
  geometry evidence: allowed

bounded context ring
  identity evidence: forbidden
  geometry assistance: allowed and separately accounted
```

- `core` 必须来自 P0 handoff region 或另行获准且有完整 lineage 的 mask provider，不能使用 evaluator truth mask。
- identity representation 不得拼接、池化或以其他方式吸收 context ring。
- context correspondence 可以帮助估计公共空间变换，但必须单列为 `context_support`。
- 没有非零且通过 gate 的 referent-core geometry 时，context-only support 只能输出
  `GEOMETRY_UNSUPPORTED_CONTEXT_ONLY`，不能使 referent `ELIGIBLE`。
- 醒目招牌、墙面或邻近门窗可以帮助空间对齐，但不能成为 referent identity shortcut。

## 5. 联合 eligibility 与诊断归因

```text
GEOMETRY_SUPPORTED AND IDENTITY_SEPARATED
  -> ELIGIBLE

GEOMETRY_SUPPORTED AND IDENTITY_AMBIGUOUS
  -> AMBIGUOUS_IDENTITY

otherwise
  -> NOT_ELIGIBLE
```

这是两种 authority 的最小联合要求，不把两条 learned RGB path 宣称为统计独立。必须分别报告 geometry 与 identity
endpoint，不能只报告 joint eligibility，因为联合值无法定位失败层。

正式 endpoint 至少包括：

1. initialization eligibility coverage，按预冻结 source 与场景 strata 报告固定 denominator；
2. referent-core geometry support、context-only rate、degeneracy 与 abstention；
3. true cross-view 对 hardest same-scene confuser 的 separation、false bind 与 abstention；
4. rotation、translation、reappearance 与 small/repetitive target 下的 cross-view support；
5. joint `ELIGIBLE / AMBIGUOUS_IDENTITY / NOT_ELIGIBLE / NOT_OBSERVABLE` accounting。

本 design 不预设数值 readiness gate。任何可执行 successor 必须在 provider 与 cohort outcome 不可见时冻结 exact gate、
cluster-aware analysis、缺失处理和 terminal adjudication；事后不得因覆盖低而降低 gate。

## 6. Candidate pool，不是已选架构

- EfficientLoFTR、LoFTR、RoMa 等可以作为 correspondence provider 候选，但本协议不选模型，也不允许 matcher
  直接获得 geometry authority。
- DINOv2、DINOv3 或 segment-level representation 可以作为 identity provider 候选，但本协议不预判其
  instance separability。
- Revisit Anything 只提供 segment/local representation 的设计启发；SAM2 最多是未来的 mask/support provider。
- MASt3R、VIO、SLAM、depth 与 world pose 不进入 P1-W2。

正式 implementation selection 应选择一个最小 matched interface，而不是建立模型 zoo。provider availability、许可、
checkpoint identity、运行资源与输出 schema 必须在 outcome-blind selection receipt 中冻结。

## 7. 数据角色与不可复用边界

- 已消费的 17 个 P1-W1 Stage A episode 只能作为 `DEVELOPMENT_DIAGNOSTIC`，可在另行授权后用于接口调通、failure
  accounting 与候选选择；不能再次承担 confirmation 或 Stage A v2 verdict。
- 一旦利用这 17 个 episode 的输出选择模型、阈值、crop、context 或 representation，所有选择均带 outcome-aware
  lineage，必须完整记录。
- 任何能力确认需要 fresh、outcome-blind、source-parent-disjoint cohort，并预先包含 small target、low texture、
  repeated facade 与 same-scene confuser。数据选择、物化和运行均未由本协议授权。
- 缺失 source authority、truth leakage、provider/interface failure 或冻结 denominator 不足时，终态必须
  `NOT_EVALUABLE`，不能记作算法负例。

## 8. 终态与后续权限

未来可执行 protocol 必须从以下 terminal family 中 outcome-blind 冻结精确定义：

```text
P1_W2_ANCHOR_INTERFACE_SIGNAL_ESTABLISHED
P1_W2_GEOMETRY_SUPPORT_LIMITED
P1_W2_IDENTITY_SEPARABILITY_LIMITED
P1_W2_RGB_REFERENT_INTERFACE_NOT_SUPPORTED
P1_W2_NOT_EVALUABLE_DATA_OR_INTERFACE
```

解释边界：

| geometry | identity | 只允许的解释 |
|---|---|---|
| sufficient | sufficient | 可另行设计 Stage A v2；不自动授权执行 |
| sufficient | limited | 下一问题是 confusable-instance identity representation |
| limited | sufficient | 下一问题是 small-target/cross-view spatial observability |
| limited | limited | 重新评估当前 RGB referent interface，不进入 tracker zoo |

`P1_W2_ANCHOR_INTERFACE_SIGNAL_ESTABLISHED` 最多授权一个新的 Stage A v2 protocol design discussion。它不自动授权
tracking、reacquisition、keyframe write、SAM2 propagation、W1-T1/Stage B、SLAM、Android/default-App 或产品/安全
claim。

## 9. 当前停止点

本文件只冻结 P1-W2 的问题、证据权限、endpoint family 与失败归因：

```text
implementation selection: not authorized
data selection/materialization: not authorized
model/checkpoint download: not authorized
execution: not authorized
Stage A v2 and later stages: not authorized
```

下一次若获授权，唯一合法动作是 outcome-blind `IMPLEMENTATION_AND_DATA_SELECTION`：选择一个最小 interface，冻结
provider identity、数值 gate、cohort specification、analysis unit、资源预算与不可复用 lineage；不得直接运行结果。
