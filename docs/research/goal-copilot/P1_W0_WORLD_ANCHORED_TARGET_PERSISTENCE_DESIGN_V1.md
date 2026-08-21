# P1-W0 World-Anchored Target Persistence design V1

状态：`DESIGN_COMPLETE / MEMORY_SEMANTICS_AND_AUTHORITY_BOUNDARIES_ONLY / NO_BASELINE_EXECUTION / NO_MODEL_SELECTION / DEFAULT_APP_UNCHANGED`

## 1. 设计问题

P1-W0 回答：

> 对于入口等 scene-fixed goal，当当前画面中没有目标 observation 时，系统如何仍然保存“它是谁、它大概在哪里、以及自己对这两件事分别有多确定”？

这里的 `World-Anchored` 指 memory 的对象是 real-world referent，而不是连续的 2D track；它不表示系统已经拥有 metric world pose、3D map 或 SLAM。空间锚点必须显式声明自己的 reference frame 和质量，不能把粗 bearing 包装成 world coordinate。

P1-A4 已建立的 consumed ADT Development 结论保持不变：强 point correspondence 可以维持几何上连续的表面，但不足以承担 physical-object identity authority。P1-A 到此关闭；W0 是抽象和 authority 的重定义，不是 A4 tracker rescue。

## 2. 范围与非目标

W0 只冻结：

- memory semantics；
- reference-frame semantics；
- identity、space、observation 与 observability 的独立有效性；
- 模块 authority boundaries；
- reacquisition invariant；
- 未来最小 evaluator 应回答的问题。

W0 不执行：

- VO、SLAM、keyframe matcher、DINO、TAPIR、flow 或其他模型选择；
- baseline、数据选择、阈值搜索、性能评测或产品状态机实现；
- Android/default-App 接入；
- metric world map、真实用户效果、安全或产品能力主张。

## 3. 目标运动类别

空间 reference frame 必须相对于目标运动类别解释：

```text
SCENE_FIXED       建筑入口、电梯、固定服务台、货架、固定座位区域
PLATFORM_FIXED    公交车门、车厢内设施等相对移动载体固定的目标
MOVING            人、车辆、工作人员等自主移动目标
UNKNOWN_MOTION    当前证据不能可靠分类
```

W0 优先设计 `SCENE_FIXED`。`PLATFORM_FIXED` 不能静默按建筑世界固定处理；`MOVING` 和 `UNKNOWN_MOTION` 不得复用 scene-fixed bearing propagation 获得 identity 或位置 authority。

## 4. 正交 memory 表示

最小逻辑表示为：

```text
TargetWorldMemory

referent_id
motion_class

ReferentIdentity
  source_keyframe_id
  source_region
  fixed_reference_evidence
  identity_quality: VALID | DEGRADED | INVALID | UNKNOWN

SpatialAnchor
  reference_frame
  anchor_keyframe_id?
  target_bearing_or_ray?
  pose_relation?
  uncertainty
  anchor_quality: GOOD | DEGRADED | STALE | INVALID | UNKNOWN

CurrentObservation
  frame_id
  candidate_region | NONE
  spatial_compatibility
  independent_identity_confirmation

referent_state
observability_reason
```

`ReferentIdentity`、`SpatialAnchor` 与 `CurrentObservation` 分别失效。下列状态都合法：

```text
identity_quality = VALID
anchor_quality = STALE
current observation = NONE
```

以及：

```text
anchor_quality = GOOD
spatial compatibility = SUPPORTED
identity confirmation = INSUFFICIENT
```

第二种情况仍不得产生 `VISIBLE_CONFIRMED` 或 `REACQUIRED`。

## 5. Reference frame 语义

`SpatialAnchor.reference_frame` 只能取：

```text
CAMERA_RELATIVE
KEYFRAME_RELATIVE
WORLD_RELATIVE
```

### CAMERA_RELATIVE

只保存相对于某一相机时刻的 bearing/ray，并利用有界 camera rotation 更新方向。它不隐含 metric translation、场景深度或固定 world point。发生超出其能力的平移、motion estimate 缺失或累计不确定性越界时，anchor 必须降级为 `STALE` 或 `UNKNOWN`。

### KEYFRAME_RELATIVE

把目标区域绑定到一个或多个场景 keyframe，并通过当前画面到 keyframe 的可验证局部关系支持 relocalization 和局部 candidate search。Keyframe match 不自动证明目标 identity，也不自动升级为 world pose。

### WORLD_RELATIVE

只有在可靠的 3D/pose/map 表示及其 uncertainty 可用时才能建立。Monocular representation 若没有 metric scale，必须明确标为 scale-ambiguous；不得仅因名称是 `WORLD_RELATIVE` 就声称 metric distance、绝对位置或稳定长程 anchoring。

Reference-frame 升级必须由新增空间证据触发；point continuity、identity similarity 或时间连续本身都不能升级 frame authority。

## 6. 最小状态与 observability

W0 不设计完整产品 FSM。最小 `referent_state` 为：

```text
UNBOUND
BOUND_NOT_OBSERVED
VISIBLE_UNCONFIRMED
VISIBLE_CONFIRMED
SPATIAL_ANCHOR_STALE
```

`SPATIAL_ANCHOR_STALE` 表示 referent memory 可以仍然存在，但当前空间预测不得继续用于方向陈述或局部确认 authority；它不把 `identity_quality` 改成 invalid。

以下值是正交的 `observability_reason`，不是更多 global states：

```text
IN_VIEW_CANDIDATE
OUT_OF_VIEW
OCCLUDED_EVIDENCED
NO_OBSERVATION
UNKNOWN
```

只有 camera/FOV geometry 足以支持时才能写 `OUT_OF_VIEW`。只有独立场景几何或明确遮挡关系足以支持时才能写 `OCCLUDED_EVIDENCED`。否则必须保持 `NO_OBSERVATION` 或 `UNKNOWN`；缺失 observation 不是 negative identity evidence。

## 7. Authority boundaries

| 组件 | 可以做什么 | 不得做什么 |
|---|---|---|
| P0 referent handoff | 建立初始 `referent_id`、source keyframe 与 region | 用后续 persistence 反向证明 P0 correctness |
| camera motion / localization | 更新 spatial relation、uncertainty 与 anchor quality | 宣告当前 candidate 是原 referent |
| keyframe memory | 限定可能重现区域和局部 search region | 把 keyframe match 等价成 object identity |
| candidate generator | 在被授权区域提出零个或多个候选 | 创建 referent、确认 identity 或伪造 observation |
| A2-style fixed-reference verifier | 提供独立的 physical-identity support/contradiction | 单独触发 reacquisition 或创建空间兼容性 |
| TAPIR / flow / correspondence | 提供短时局部 bearing/region continuity | 承担 physical identity authority 或跨失效锚点延续 bbox |
| memory fusion | 组合独立空间与身份 evidence，并保守降级 | 用任一单通道 evidence 替代双条件确认 |

## 8. 不可违反的 invariants

### Reacquisition

```text
REACQUIRED
=
spatial compatibility
AND
independent identity confirmation
```

两项必须来自职责独立的 evidence path。空间预测很明确但 identity evidence 不足，或视觉 identity 很像但空间不一致，都必须拒绝 `REACQUIRED`。

### Observation honesty

```text
CurrentObservation.candidate_region = NONE
=>
不得输出或延续当前 bbox
```

Memory 可以在 observation 为 `NONE` 时继续存在；保留 memory 不等于伪造当前 observation。

### Uncertainty monotonicity

没有新增支持证据时，累计 motion、时间和 frame mismatch 只能保持或降低 anchor quality，不能自动恢复。超过当前 tier 能力的运动必须产生 `SPATIAL_ANCHOR_STALE / UNKNOWN`，不能继续给出旧方向。

### Independent invalidation

Anchor stale 不删除仍有效的 identity memory；identity invalid 不因空间预测明确而恢复。`UNKNOWN` 不得静默转换成 negative、confirmed 或 visible。

## 9. 分层能力边界

```text
Tier 0
CAMERA_RELATIVE bearing persistence
有界相机旋转；明显平移或运动不可观测时 fail stale

Tier 1
KEYFRAME_RELATIVE relocalization
回到相似视角后做局部 candidate generation + independent identity confirmation

Tier 2
WORLD_RELATIVE monocular 3D / SLAM anchoring
只有 Tier 0/1 的产品问题确实不能覆盖且新增复杂度有明确收益时才考虑
```

Tier 0/1 的目标是验证轻量 world-referent memory 是否已覆盖大量“刚才那个门去哪了”的交互，不以完整 metric map 为默认终点。

## 10. 未来最小 evaluator 合同

W0 不运行 evaluator，也不冻结数据集或数值 admission threshold；只冻结 evaluator 的问题类型。未来极小 baseline 至少必须包含以下可判定场景：

| 场景 | 正确行为 |
|---|---|
| 转头使 scene-fixed target 离开视野 | bearing 传播到正确侧；observation 为 `NONE`；不输出 bbox |
| 转回相似视角且候选空间与身份均相容 | 可以产生 `REACQUIRED` 事件并进入 `VISIBLE_CONFIRMED` |
| spatial prediction 相容、identity 不相容 | 拒绝 reacquisition |
| identity 相似、spatial prediction 不相容 | 拒绝 reacquisition |
| 无当前 observation | memory 可保持，但 fabricated bbox count 必须为 0 |
| identity memory valid、anchor stale | 保留 referent identity；停止方向陈述和 spatially authorized search |
| 明显平移超过 Tier-0 能力 | `SPATIAL_ANCHOR_STALE / UNKNOWN`，不得继续声称旧 bearing 有效 |
| motion/localization evidence 不足 | 保守降级，不把缺失证据解释成目标消失或 identity negative |

未来 evaluator 的主量应围绕：

- anchor 有效条件下的 bearing side/angular compatibility；
- `NONE observation` 期间 fabricated observation/bbox 数；
- 单通道错误触发的 false reacquisition 数；
- stale/unknown 降级是否及时；
- local re-entry candidate 上的 identity-confirmed reacquisition。

它不以逐帧 tracking accuracy 或 box continuity 作为主要目标。

## 11. 当前终点与唯一后继边界

P1-W0 到设计终点即停止。当前不选择实现、不运行 baseline、不接 App。

若用户另行授权，唯一合理后继是一个极小、可逆的 `P1-W1 Tier-0/Tier-1 baseline protocol`：先验证 camera-relative bearing、keyframe-relative local relocalization、双条件 reacquisition 与 stale downgrade；不得自动扩张为 SLAM、模型 zoo、fresh cohort 或产品 FSM。

Claim ceiling：`ARCHITECTURE_DESIGN_ONLY / NO_EMPIRICAL_CAPABILITY / NO_PRODUCT_OR_SAFETY_AUTHORITY`。
