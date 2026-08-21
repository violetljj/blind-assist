# P0 Goal Grounding Protocol V1

状态：`FROZEN_MECHANICS_CONTRACT / GOAL_REFERENCE_SET_ADDENDUM / NAMED_BUILDING_ENTRANCE_GROUNDING / PASSIVE_OBSERVATION_ONLY / MOCK_VALIDATION_ONLY / NO_BASELINE / NO_SKY / DEFAULT_APP_UNCHANGED`

协议 ID：`BA-P0-NAMED-BUILDING-ENTRANCE-GROUNDING-V1`

2026-08-21 prospective goal-reference addendum：以下 `UNIQUE / SET_VALUED / AMBIGUOUS` 语义只适用于新增
episode，不回写 addendum 前的 mechanics receipt 或 P0-S1 终态。

## 研究问题

给定一个目标描述和一段冻结的第一视角被动观察窗口，系统能否将目标绑定到任一合法可见 referent 和原始帧
空间区域；当语言不足以确定合法 referent 集、目标不在场、观测无效或证据不足时，能否拒绝强猜？

V1 只覆盖 `Named Building Entrance Grounding`：用户指定一个有名称的建筑，系统寻找属于该建筑的
入口。它不是通用生活目标 benchmark，也不同时覆盖公交车门、空座位、电梯按钮、商品、收银台或
网约车。

## 当前授权

本协议当前只授权合同、schema、deterministic evaluator mechanics、mock fixtures 和 evaluator unit tests。

明确禁止：

- 下载、采集、生成或物化新 cohort；
- 集成或运行 detector、OCR、VLM、SAM、open-vocabulary model 或其他模型；
- 实现完整 Copilot Brain；
- 实现 P1 Persistence、P2 Approach/Completion 或 P3 Active Perception；
- 让系统请求用户转头、扫描、靠近、抬头或改变摄像机位置；
- 调用 Sky 或修改其 search surface；
- 修改 Android/default-App、guidance 或 safety 产品链路；
- 冻结 baseline 数值门或产生科学/产品结论。

## 固定能力边界

P0 是目标驱动、实例级、空间锚定且允许拒绝的 grounding：

1. 目标驱动：输入是“XX 医院入口”，不是“检测所有门”；
2. 实例级：普通入口或相邻建筑入口不能替代目标实例；
3. 空间锚定：`GROUNDED` 必须指向一个 source frame 中的具体 normalized XYXY region；
4. 证据可追溯：每个 Brain 决策必须回溯到明确的 `evidence_id`；
5. 允许拒绝：候选缺失、歧义、目标不在场或观测无效不能被强行改写成目标不存在或已找到；
6. 异步安全：slow evidence 永远绑定它处理的原始帧；P0 不声称把旧坐标追踪到了当前帧。

P0 不负责 tracking、reacquisition、approach、arrival、passability、active view selection、navigation
safety 或“可以通过”的输出。

## 输入合同

每个 episode 由 `p0_episode_schema.json` 验证，至少包含：

```text
episode_id
goal_spec
observation_window
observation_valid
target_visible
goal_reference_resolution
valid_target_instances
acceptable_spatial_regions
distractor_instances
target_min_side_px
visibility_fraction
text_support
scene_condition
grounding_expectation
```

`grounding_expectation` 只能是：

- `MUST_GROUND`：证据角色要求系统绑定正确实例；
- `MUST_BE_AMBIGUOUS`：当前语言/上下文不足以可靠确定合法 referent 集；
- `MUST_ABSTAIN`：目标不在场或证据不足；
- `INVALID_OBSERVATION`：输入窗口本身不可评估。

Goal reference truth 显式区分：

- `UNIQUE`：`valid_target_instances` 恰有一个物理目标；
- `SET_VALUED`：至少两个物理目标都满足当前指令，选中其中任何一个都不得判错；
- `AMBIGUOUS`：现有语言/上下文不足以可靠确定合法目标集合，不得预设单一 bbox，系统返回 `AMBIGUOUS`
  或 fail-closed abstention 均为正确处理。

`acceptable_spatial_regions` 必须严格等于所有 valid target 在 observation window 中 regions 的并集；不存在
额外隐藏的 single-target region。`target_visible` 只表示至少一个已建立的 valid referent 在窗口中有 region。
因此 `AMBIGUOUS` 不进入 Provider recall 或 exact Brain-selection 分母。这里不实现 clarification、对话历史、
路线偏好或最近入口策略。

Mock fixture 是 evaluator mechanics，不是 scientific cohort。真实 cohort 的来源、去重、场景数量、分层
分母、Development/fresh 角色和缺失数据规则均未在 V1 中授权。

## 输出合同

系统输出由 `p0_output_schema.json` 验证，包含 Provider run records、evidence ledger、candidate manifest 与
一个 `GroundingDecision`。

```text
GroundingDecision
├─ status
│  ├─ GROUNDED
│  ├─ AMBIGUOUS
│  ├─ ABSTAIN_NO_RELIABLE_EVIDENCE
│  └─ INVALID_OBSERVATION
├─ selected_candidate_id
├─ ranked_candidate_ids
├─ source_frame_id
├─ decision_timestamp_ms
├─ spatial_region
├─ goal_identity_support
├─ spatial_support
├─ confidence
├─ supporting_evidence_ids
├─ competing_candidate_ids
├─ abstention_reason
└─ persistence_handoff_token
```

`GROUNDED` 的必要条件：

- `selected_candidate_id`、`source_frame_id`、`spatial_region`、`confidence` 和 handoff token 非空；
- `goal_identity_support == SUPPORTED` 且 `spatial_support == SUPPORTED`；
- supporting evidence 非空、存在、有效、未过期并与候选绑定；
- 至少一项 supporting evidence 的 identity claim 精确匹配 `goal_spec.target_name`；
- handoff token 与 selected candidate、source frame、region 和 evidence IDs 一致。

非 `GROUNDED` 状态不得携带 selected candidate、spatial region 或 persistence handoff token。

每条异步 evidence 至少携带：

```text
provider_id
evidence_id
evidence_type
source_frame_id
source_timestamp_ms
region_in_source_frame
confidence
validity
expiry_timestamp_ms
provenance
```

`NOT_RUN`、`RUN_FAILED`、`INVALID_OUTPUT` 与“运行成功但没有候选”必须保持可区分。`NO_CANDIDATE` 不能
解释为目标不存在。Slow evidence 返回较晚时仍只锚定 `source_frame_id`；若它被标为 `EXPIRED` 或超过
expiry，不能支撑 `GROUNDED`。

## Evaluator 分层

### Layer 1 — Evidence Provider Availability

先判断正确 spatial target 是否进入任何成功运行 Provider 的候选集合，再记录 identity support 是否可用：

```text
correct_candidate_available
provider_recall_at_k (K = 1, 3, 5)
correct_candidate_rank
correct_candidate_provider_ids
goal_identity_evidence_available
provider_run_statuses
provider_failure_classes
target_min_side_px
visibility_fraction
scene strata
```

Availability 使用候选 region 与全部 `acceptable_spatial_regions` 的最大 IoU；`SET_VALUED` 下命中任一 valid
target region 都算 correct。V1 mechanics threshold 固定为
`0.5`，只用于验证 evaluator，不是 baseline admission 数值门。

### Layer 2 — Brain Selection / Fusion

只有目标在场、观测有效且 correct candidate available 时，selection 指标才可识别：

```text
identifiability = IDENTIFIABLE | NOT_IDENTIFIABLE
top1_correct_given_available
wrong_instance_given_available
correct_abstention_under_ambiguity
candidate_rank_improvement
identity_match
spatial_match
stale_evidence_used
```

correct candidate 不存在时，selection fields 必须为 `null`，不能记成 Brain selection failure。

### Layer 3 — End-to-End P0 Grounding

逐 episode 输出可归因 outcome code，而不是只给平均准确率：

```text
CORRECT_GROUNDING
CORRECT_AMBIGUITY
CORRECT_ABSTENTION
INVALID_OBSERVATION_HANDLED
PROVIDER_CORRECT_CANDIDATE_UNAVAILABLE
WRONG_INSTANCE
SPATIAL_LOCALIZATION_ERROR
GOAL_IDENTITY_ERROR
FALSE_GROUNDING_TARGET_ABSENT
MISSED_REQUIRED_GROUNDING
STALE_EVIDENCE_USED
INVALID_SYSTEM_OUTPUT
```

Batch summary 至少按 target size、visibility、single/multiple entrance、text support、same-class distractor、
target presence、illumination 与 view angle 分层。所有 rate 都携带显式 numerator/denominator；分母为零时
value 为 `null`，不补 0 或 1。

### P1 handoff mechanics

P0 不实现 persistence，但必须验证 handoff 可交接：token 绑定 selected candidate、source frame、source
region 与 supporting evidence。它不声称 region 已经传播到当前帧，也不产生 tracking success 指标。

## Mock mechanics cases

单元测试至少覆盖：

1. correct candidate 存在并被正确选中；
2. correct candidate 存在但选择 wrong instance；
3. correct candidate 完全不存在；
4. target absent 时强行 grounding；
5. 多候选时正确保持 `AMBIGUOUS`；
6. slow evidence 已过期但仍被用于 grounding；
7. region 正确但 goal identity 错误；
8. goal identity 正确但 region 错误；
9. invalid observation 被正确拒绝；
10. handoff token 与 decision 漂移时 fail closed。

## Baseline 前顺序

1. 本 V1 mechanics contract 与 schemas；
2. mock manifests 与 evaluator tests；
3. 另行冻结真实 cohort materialization 规则与数据角色；
4. cohort 冻结后、任何模型或 baseline outcome 前冻结数值门；
5. 用户另行授权后才运行 baseline，且必须同时输出 Provider candidate manifest 与 Brain decision trace。

## Claim ceiling

V1 通过最多证明 P0 合同和 evaluator mechanics 能区分 provider unavailable、selection error、identity
error、spatial error、stale evidence、correct abstention 与 handoff drift。它不证明任何模型、数据集、
Goal Grounding 能力、真实用户效果、连续引导、安全性或产品可用性。
