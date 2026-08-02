# 裁判审计 R0：primitive truth first，再派生 actionability

机器可读合同：[JUDGE_AUDIT_R0_CONTRACT_2026-08-02.json](JUDGE_AUDIT_R0_CONTRACT_2026-08-02.json)。实现：[judge_audit.py](../../../scripts/research/eval_validity_r0/judge_audit.py)。

当前正式合同仍是 `PRE_OUTPUT_LOCKED`。现有 78 个 source-mask candidate 不能直接进入正式盲审；先执行不计入正式分母的 `CALIBRATION_BURNED` pilot。

## 核心修订

reviewer 不再直接提交 `ACTIONABLE/NON_ACTIONABLE` 或 `event_reminder_now`。每个 reviewer 只对每个源帧提交六类可观察 primitive：

```text
visibility:       EVALUABLE | NOT_EVALUABLE
path_relation:    BLOCKING_PATH | NON_BLOCKING_PATH | AMBIGUOUS
motion_relation:  APPROACHING | LATERAL_PASS | RECEDING | STATIC_OR_UNCLEAR
phase:            BEFORE_INTRUSION | CURRENT_INTRUSION | PASSED_CLEAR | UNKNOWN
route_certainty:  SINGLE_PLAUSIBLE_ROUTE | MULTIPLE_PLAUSIBLE_ROUTES | UNKNOWN
evidence_quality: CLEAR | BLUR | OCCLUSION | CAMERA_ROTATION | INSUFFICIENT
```

### Primitive 操作定义与字段级证据边界

当前 primitive policy 版本为 `primitive_observability_v4`。所有几何字段先固定同一个
`route_anchor`：**当前承载相机的行人支撑面（sidewalk/path/trail 或其他当前可行走支撑面）及其可见的正前方连续延伸**。
不得把未与当前支撑面连接的平行/背景步道、相机未占用的车道或假设中的绕行路线算进来；只有当前支撑面明确连接到两个以上实质不同的可行走延伸，才记录
`MULTIPLE_PLAUSIBLE_ROUTES`。

`path_relation` 只问当前帧的物理关系：

- `BLOCKING_PATH`：可定位的物理区域/边缘与当前 route corridor 相交，或把可见的连续通行宽度完全封住；不要求知道目标类别或运动状态。
- `NON_BLOCKING_PATH`：当前 corridor 清楚且无占用，或局部区域在 corridor 外，或能看见连续可通行的绕行宽度。
- `AMBIGUOUS`：当前路线锚点、区域关系、相关边缘或通行宽度无法定位；不能因为类别未知、提醒决策不确定或没有单独目标就使用 `AMBIGUOUS`。

`route_certainty` 只问当前支撑面的正前方延伸：一个明确连续延伸为
`SINGLE_PLAUSIBLE_ROUTE`，两个以上实质不同且相连的可行走延伸为
`MULTIPLE_PLAUSIBLE_ROUTES`，当前支撑面/延伸本身无法定位才为 `UNKNOWN`。

`evidence_quality` 不是“画面好不好看”，而是当前帧是否足以解释上述几何关系：
只有运动/对焦模糊实质遮住相关边界才记 `BLUR`，前景/场景遮挡实质隐藏相关区域才记
`OCCLUSION`，全局旋转使路线方向无法稳定解释才记 `CAMERA_ROTATION`；轻微软化和普通阴影仍为
`CLEAR`；没有可用路线锚点才记 `INSUFFICIENT`。判定优先级固定为
`INSUFFICIENT → CAMERA_ROTATION → OCCLUSION → BLUR → CLEAR`。

`phase` 使用版本 `phase_observability_v2`，只描述当前路线占用相对于允许观察前缀的时相，不能由
actionability 反推：当前 `NON_BLOCKING_PATH` 且允许前缀中没有更早的 blocking path 为
`BEFORE_INTRUSION`；当前 `BLOCKING_PATH` 为 `CURRENT_INTRUSION`；当前
`NON_BLOCKING_PATH` 且允许前缀中观察到更早的 blocking path 为 `PASSED_CLEAR`。当前 path
关系、路线锚点、证据质量或允许前缀不足以确定占用状态时才为 `UNKNOWN`。causal reviewer 只能使用当前帧和过去前缀，不能用未来帧把当前 clear 改成 passed 或预判未来侵入；retrospective reviewer 可描述全事件顺序，但不能改写 current-only 的 visibility/path/route/evidence 字段。

字段证据窗口如下，review packet 同时给出对应的单帧引用，防止把 temporal 视图误用于几何字段：

```text
visibility / path_relation / route_certainty / evidence_quality：CURRENT_RGB_FRAME_ONLY
motion_relation / phase（causal）：CURRENT_PLUS_PAST_PREFIX
motion_relation / phase（retrospective）：FULL_EVENT_RGB
```

### `visibility` 的操作定义与证据边界

`visibility` 使用版本 `visibility_observability_v2`，只回答一个当前帧问题：

```text
当前 RGB 帧中，前方路线场景是否存在可定位的可视区域？
```

- `EVALUABLE`：当前帧能定位前方路线、地面/边界或场景区域；不要求识别目标类别，也不要求该区域存在障碍或已经判断是否挡路。
- `NOT_EVALUABLE`：当前帧没有可定位的路线/场景区域、路线场景完全出框，或路线场景被完全遮挡。
- `BLUR`、`OCCLUSION`、`CAMERA_ROTATION` 和“无法判断 path/phase/route”不能自动改写 `visibility`；它们分别留在 `evidence_quality` 或对应 primitive。
- `visibility` 不得由 `path_relation`、`motion_relation`、`phase`、`route_certainty`、`evidence_quality` 或 `actionability` 反推。没有障碍的宽通道仍可以是 `EVALUABLE`。

证据窗口被逐字段冻结：

```text
visibility / path_relation / route_certainty / evidence_quality：CURRENT_RGB_FRAME_ONLY
motion_relation / phase（causal）：CURRENT_PLUS_PAST_PREFIX
motion_relation / phase（retrospective）：FULL_EVENT_RGB
```

因此，未来帧不能把当前缺失的路线场景“补成”可见；retrospective 只能在其自身报告视图中使用未来信息，不能裁决 causal truth。packet 必须同时提供
`current_rgb_frame`、各几何字段的 current-only frame cards 和 `temporal_rgb_frames`，review 与 seal 必须记录上述窗口。

输出盲的 host-side frozen rule 再派生：

```text
ACTIONABLE_NOW = YES
```

仅当 visibility、path、phase、route 和 evidence 均满足冻结条件；任何未解析 primitive 都派生为 `UNKNOWN`。`cleared`、onset 和 clear boundary 同样由 primitive 序列和固定时间戳规则派生。review packet 禁止含直接 action label。

reviewer 只看到 opaque `review_item_id`。候选类别、source-session、selection reason、pair role、YOLO 命中与任何模型/oracle 输出均隐藏；`blocking`、`parallel`、`negative`、`approach`、`curb`、`unknown-object`、`roadside` 等语义词不能出现在 reviewer item ID 中。

review packet 必须先完成两名 causal 与一名 retrospective 的独立提交并封存；每份 review
带 `sealed_before_pair_selection=true`。封存 hash 绑定到后续 pair manifest，防止看到标签后
回改 review 或重新挑 pair。

## 数据合同

正式 cohort 为 50–100 个 parent event、每 session 一个 event。每类最低 6 个事件、至少 3 个 source-session；任一 session 不得超过总事件的 10%。八类是：已知正前方障碍、YOLO 未知对象、路边不阻塞、正面接近、横向经过、纯相机运动、宽通道、证据不足。`insufficient_evidence` 不能同时充当其他类别配额。

每个事件至少 20 个连续帧且连续时长至少 3 秒。正事件必须冻结至少 0.8 秒 `BEFORE_INTRUSION`、0.8 秒 `CURRENT_INTRUSION`、0.8 秒 `PASSED_CLEAR`。仅有 20 帧但不足 3 秒，或 onset 左截断的事件，不能进入正式分母。

每个 event 还必须记录一个不向 reviewer 暴露的 `discovery_arm`。允许的 arm 为
`source_mask`、`random_continuous_rgb`、`motion_temporal_change`、`metadata_only_normal`。
burned pilot 可以暂时只有一个 arm；正式 cohort 至少要有 `source_mask` 加一个独立 arm，
且所有 arm 进入相同 RGB 盲审。候选来源只作为审计分析字段，不能进入 reviewer packet。

正式 cohort 的相似框反事实目标为 8–12 对；burned calibration pilot 只冻结 3–4 对。pair 构造有严格时序：

```text
output-blind discovery 冻结 pilot events
→ 2 causal + 1 retrospective primitive reviews
→ 封存 reviews 与 bundle hash
→ YOLO 只以 selection-only 身份读取 box/尺度/位置与预冻结的 `selection_time_slot`
→ deterministic 枚举全部 eligible pairs，按预冻结排序冻结 pair 数量与 pair manifest
→ 打开 labels，运行物理反事实测试
```

pair manifest 必须声明 `stage=AFTER_REVIEWS_SEALED`、`yolo_role=SELECTION_ONLY`、
`yolo_visible_to_reviewers=false`、`yolo_used_for_truth=false`、primitive/derived labels
对 pair-builder 均不可见，并绑定封存 review hash。pair-builder 只能读取并记录
`selection_time_slot`（固定采样 slot，或独立的 frame index/timestamp），不能读取
`reviewed_event_phase`、`reviewed_motion_relation` 或任何 actionability/physical-condition
字段。候选 pair universe 的 hash、数量与 deterministic ordering receipt 也必须封存。
每对除 YOLO 框相似度 `>=0.90` 外，还要冻结相近距离/尺度、相近位置、相近可见性和相同
`selection_time_slot`；打开 labels 后才检查 reviewer primitive 的 phase/motion 是否匹配，
以及 path/actionability 是否形成物理反差。若冻结后有效 pair 不足各自模式的冻结区间，终态是
`NOT_EVALUABLE/HOLD`，不得回头按标签挑样本。若 primitive 或派生 actionability 是
`UNKNOWN`，该 pair 记录为 `NOT_EVALUABLE_PAIR`。

## 两条 oracle 路径

### 系统统一链

```text
current YOLO / oracle input
→ declared adapter
→ same decision kernel
→ same risk config, clock, reset and feedback
→ alert/event metrics
```

它回答“在当前 App 架构中，这种输入能否改善提醒”。每条 trace 都必须携带
`eligible_for_native_task`、`eligible_for_system_chain`、`required_inputs`、
`expected_improvement_dimension` 和 `not_evaluable_reason`。每个 oracle 只在其预先冻结的
eligible event、required inputs 和 expected improvement dimension 上比较。

### 原生信息上限链

```text
truth mask       → corridor occupancy / blocking discrimination
truth depth      → clearance ordering
truth geometry   → corridor occupancy
truth trajectory → future path intersection
```

它不经过 YOLO-shaped decision kernel，直接与物理 primitive/reference 比较。若原生信息链通过、系统统一链却没有在 eligible opportunity 上改善，终态为：

`FLAG_EVALUATION_STACK_CEILING_SUSPECTED`

嫌疑对象是 adapter、decision kernel、event policy 和 metric 的整体，不再武断命名为 metric ceiling。

## 盲审稳定性

至少两名 causal reviewer 加一名 retrospective reviewer。retrospective 只用于测量未来信息对判断的影响，不能用来裁决 causal 分歧，也不能多数票修补真值；它不参与任何通过/失败门。报告分成两层：

### Primitive construct stability

- `visibility`（当前帧场景可观察性）一致率与 classwise agreement；
- `path_relation` 一致率；
- `motion_relation` 一致率；
- `phase` 一致率；
- `route_certainty`（通道/clearance 关系）一致率；
- `evidence_quality` 与 UNKNOWN 一致率；
- boundary timing 的两帧容差一致率；
- primitive disagreement source。

### Derived policy stability

- frame/event-level `ACTIONABLE/NON_ACTIONABLE/UNKNOWN` 一致率；
- `event_cleared` 一致率；
- primitive disagreement 向 actionability 的传播比例；
- 同样 primitive 输入是否始终得到同样派生结果；
- causal 与 retrospective 的 action reversal rate，以及 `causal UNKNOWN → retrospective KNOWN`。

只有 causal A/B 的 primitive 与 derived 稳定性参与稳定性门；retrospective 结果只报告，不 adjudicate。

同时仍报告：

- causal A vs causal B 的 event/boundary consistency；
- causal consensus vs retrospective；
- causal label reversal rate；
- causal 与 retrospective UNKNOWN 比例。

## Burned calibration pilot

正式 cohort 前只允许 8–12 个事件、3–4 对反事实、两 causal 加一 retrospective。pilot 的执行顺序必须是：先冻结 output-blind events，再封存 reviews，之后才用 YOLO selection-only 构造并冻结 pair，最后生成两条 oracle trace 并运行四项审计。pilot 用于验证 packet 元数据盲化、primitive/derived 双稳定性、native information ceiling 与 kernel conversion；其事件、prompt、grid、标签规则和探针数据永久标记 `CALIBRATION_BURNED`，不得进入正式分母。

当前 50 帧 RGB/mask 探针 `rgb-probe-wz9-v1` 已明确 burned；它只证明下载/对齐/哈希链路，不能提供事件真值。

## 终态

- `HOLD_JUDGE_AUDIT_COHORT`：正式事件/覆盖/时间/来源门未冻结。
- `STOP_JUDGE_AUDIT_FAILED`：primitive truth 泄漏、物理反事实不敏感或盲审稳定性失败。
- `NOT_EVALUABLE_JUDGE_AUDIT_INPUTS`：review、matched pair、native oracle opportunity 或统一 trace 缺失。
- `FLAG_EVALUATION_STACK_CEILING_SUSPECTED`：原生信息链能区分物理任务，但系统栈没有转换为改善。
- `VALID_BURNED_CALIBRATION_PILOT`：只允许修订 packet/合同，不能作为正式结果。
- `VALID_JUDGE_AUDIT_CONSTRUCT`：仅表示正式 Development 评价构造通过，不授权训练、产品、安全、Android 或默认 App。
