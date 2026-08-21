# P1-W1 minimal world-referent baseline protocol V1

状态：`PROTOCOL_FROZEN / REVERSIBLE_EXPLORATION / IMPLEMENTATION_NOT_SELECTED / DATA_NOT_SELECTED / NO_EXECUTION / DEFAULT_APP_UNCHANGED`

Claim ceiling：`PROTOCOL_DESIGN_ONLY / NO_EMPIRICAL_CAPABILITY / NO_PRODUCT_OR_SAFETY_AUTHORITY`

## 1. 唯一问题

P1-W1 只回答：

> 加入最小空间锚点后，能否比纯 camera-relative persistence 更诚实地跨越短时不可见和相机运动，同时不增加 false reacquisition？

W1 不证明 SLAM 一般有效，不建立完整 world model，也不以 bbox 存活时间为成功标准。它检验的是 W0 的 `real-world referent belief` 抽象是否比连续 2D tracking 更适合承担 BlindAssist persistence authority。

## 2. 与 W0 和 P1-A 的边界

- [`P1-W0`](P1_W0_WORLD_ANCHORED_TARGET_PERSISTENCE_DESIGN_V1.md) 保持封口，不因 W1 改写 memory semantics。
- W1 内的 `W1-T0 / W1-T1` 是本协议的实验层级命名，不重命名 W0 的 capability ladder。
- P1-A0–A4 保持 closed。TAPIR、flow 或其他 correspondence 以后只能作为 observation provider，不能恢复 referent authority。
- A2-style evidence 只能作为各 arm 共享的 independent identity verifier；W1 不重新搜索 A2 threshold 或把历史 Development signal 升格。
- W1 只处理 `SCENE_FIXED`。`PLATFORM_FIXED / MOVING / UNKNOWN_MOTION` 不进入主指标。

## 3. 三个 matched arms

W1 只有一个 control 和两个逐级增加空间信息的 baseline。所有 arm 使用相同输入帧、P0 handoff、candidate generator family、identity verifier、confirmation rule、输出 schema 与 evaluator。

### C0 — honest camera-relative control

```text
last confirmed observation
+ bounded camera-relative rotation propagation
+ W0 fail-closed memory semantics
```

C0 可以在有界旋转内维护 bearing。明显 translation、motion 不可观测或累计 uncertainty 越界时必须 `SPATIAL_ANCHOR_STALE`。C0 不使用 keyframe relocalization、world pose 或持续 bbox。

### W1-T0 — keyframe-relative baseline

```text
last confirmed observation
+ reference keyframe
+ current-to-keyframe relative motion / feature geometry
→ bearing or coarse spatial compatibility
```

T0 只允许：

- 短时不可见或遮挡；
- 小幅转头和旋转；
- 回到相似视角后的局部 candidate region；
- keyframe-relative bearing compatibility；
- 几何退化或明显 translation 时 fail stale。

T0 禁止 SLAM、persistent 3D map、semantic/object map、全局回环数据库或跨场景 global search。Keyframe match 不能单独证明 identity。

### W1-T1 — minimal world-relative baseline

```text
camera pose source
+ target direction / coarse spatial anchor in the same gauge
+ independent identity evidence
```

T1 只比 T0 增加一个可审计的 pose/anchor interface：

```text
PoseEstimate
  frame_id
  pose_in_shared_frame
  gauge_or_scale_semantics
  uncertainty
  quality

CoarseWorldAnchor
  shared_frame_id
  direction
  bounded inverse-depth / position evidence if available
  uncertainty
  quality
```

Camera pose 加一条初始 image ray 并不足以在 translation 后唯一确定新 bearing。T1 若没有与 pose 同 gauge 的 bounded inverse-depth、三角化位置或等价空间 evidence，必须保持 `STALE / UNKNOWN`，不得伪造 world-relative compatibility。

T1 可以使用 outcome-blind 选定的 monocular VIO 或 visual SLAM pose source，但本协议不选择实现。它禁止 semantic map、object map、NeRF、Gaussian Splatting、persistent scene graph、复杂 reconstruction 和以 object identity 修补 pose。

## 4. 固定 referent output contract

每个 arm 每帧只输出：

```text
ReferentSnapshot
  referent_id
  identity_state
  spatial_anchor_state
  observation_state
  observability_reason
  reference_frame
  candidate_region | NONE
  bearing_estimate | NONE
  bearing_uncertainty | NONE
  spatial_compatibility
  independent_identity_confirmation
  reacquisition_status
  directional_guidance_authorized
```

以下 W0 invariants 原样继承：

```text
REACQUIRED
=
spatial compatibility
AND
independent identity confirmation

candidate_region = NONE
=>
不得输出或延续 bbox

spatial_anchor_state = STALE
=>
directional_guidance_authorized = false
```

Identity、anchor 和 observation 分别失效。Arm 不得通过多报 `UNKNOWN` 删除 referent memory，也不得通过延续 bbox 隐藏 observation 缺失。

## 5. 最小场景支持门

未来 execution roster 必须在 outcome access 前固定，并至少包含以下非零机会；任一为零则对应 endpoint `NOT_EVALUABLE_DATA_SUPPORT`，不能把缺失场景计为成功：

| 场景桶 | 必须观察的机制 |
|---|---|
| rotation out-of-view and return | bearing side、honest `NONE`、local return |
| short occlusion | memory retained without fabricated observation |
| similar-view re-entry | spatial + identity 双条件 reacquisition |
| spatial-compatible identity distractor | identity veto |
| identity-similar spatial mismatch | spatial veto |
| geometry-degenerate keyframe | timely T0 stale |
| translation beyond C0/T0 capability | stale downgrade and zero stale guidance |
| translation with valid shared-frame anchor | T1 incremental compatibility opportunity |

同一 episode 可以贡献多个桶，但 aggregate 与每桶分母必须同时报告，不能用易旋转场景淹没 translation failure。

## 6. Truth firewall 与因果隔离

系统侧只接收所选 RGB-derived inputs 和该 arm 明确允许的 pose/geometry output。Evaluator-private physical target identity、visibility、camera/target geometry 与 event labels 不得进入 candidate generation、identity verification、anchor update 或 state transition。

必须保持：

- post-initialization GT reads 为 0；
- future-frame reads 为 0；
- GT reset、oracle bbox、oracle pose、semantic identity 和 global target search 为 0；
- C0/T0/T1 使用同一 P0 source referent、相同 identity verifier 和相同 confirmation rule；
- 除空间 representation 外不允许 arm-specific threshold、candidate model 或 tracker tuning；
- 每个 arm 独立 state，不共享未来 arm 的 pose、candidate 或 confirmation output。

若 selected pose source 的接口、许可、因果性或 shared-frame semantics 不满足合同，对应 arm 为 `NOT_EVALUABLE_INTERFACE`，不得换成 evaluator pose 强行出结果。

## 7. Primary endpoints

W1 不使用 IoU、tracking survival 或 bbox continuity 作为 primary endpoint。它们若记录，只能作为 diagnostic。

### Safety/authority endpoints

```text
false_continuity
  当前 observation 被断言为原 referent，但 evaluator 显示是 background、其他实例或无受支持 observation

false_reacquisition
  REACQUIRED 事件未对应原 physical referent 的有效重现

fabricated_observation
  candidate_region/bbox 在 arm 自己声明 observation=NONE 时仍被输出或延续

stale_anchor_guidance_use
  anchor 为 STALE/INVALID/UNKNOWN 时仍授权方向性引导

single_channel_reacquisition
  缺 spatial compatibility 或 independent identity confirmation 任一条件仍触发 REACQUIRED
```

### Honesty/uncertainty endpoints

```text
honest_none_observation
  当前无受支持 target observation 时明确输出 NONE 且保留合法 memory

timely_anchor_stale
  超出 arm 能力或 uncertainty envelope 后，在预冻结 deadline 内降级

bearing_error_or_compatibility
  仅在 arm 声称 anchor 可用且 evaluator truth 可用的帧上计分
```

### Utility endpoints

```text
identity_confirmed_reacquisition
  spatial 与独立 identity 双条件成立的正确重捕获

usable_anchor_coverage
  在不违反 authority gates 的前提下可合法给出 bearing/compatibility 的机会覆盖
```

所有 coverage 必须带 denominator。Always-`NONE` 可以安全但不能建立 utility；always-continuous 可以提高 bbox survival 但会被 false continuity 和 authority gates 拒绝。

## 8. Hard gates 与比较规则

每个 arm 都必须满足：

```text
fabricated_observation = 0
single_channel_reacquisition = 0
stale_anchor_guidance_use = 0
post-initialization truth leakage = 0
future-frame access = 0
```

### Stage A：C0 vs W1-T0

T0 只有同时满足以下条件才建立信号：

1. 全部 hard gates 通过；
2. `false_reacquisition(T0) <= false_reacquisition(C0)`；
3. `false_continuity(T0) < false_continuity(C0)`，或在相同零 false-continuity 下严格提高 usable anchor coverage；
4. `identity_confirmed_reacquisition(T0) >= C0`；
5. keyframe-degenerate 与 translation-overreach 桶均按 deadline 正确 stale；
6. supported-anchor 帧的 bearing compatibility 不劣于 C0。

如果改善只来自更高 abstention，而 correct reacquisition 和 usable anchor coverage 同时下降，则终态为 `HONESTY_GAIN_ONLY_BY_ABSTENTION`，不建立 world-referent utility signal。

### Stage B：W1-T0 vs W1-T1

Stage B 不自动执行。只有 Stage A 先建立信号、translation support 非零且用户另行授权后才可进入。T1 只有同时满足以下条件才建立增量信号：

1. 全部 hard gates 通过；
2. translation subset 的 usable anchor coverage 严格高于 T0；
3. `false_reacquisition(T1) <= false_reacquisition(T0)`；
4. `false_continuity(T1) <= false_continuity(T0)`；
5. `identity_confirmed_reacquisition(T1) >= T0`；
6. pose/anchor shared-gauge 不足时正确 stale，而不是继续旧 bearing。

Exact roster、deadline、bearing tolerance 与统计 margin 必须在 implementation/data selection 后、任何 performance outcome 前冻结；不得根据结果补门或改 denominator。

## 9. 顺序、停止条件与终态

当前只冻结协议，不授权任何 stage：

```text
W1 protocol
  ↓ separate authorization
outcome-blind C0/T0 implementation + data selection
  ↓ freeze exact thresholds/roster
Stage A execution once
  ↓ only if signal + translation support + separate authorization
outcome-blind T1 pose-source selection
  ↓
Stage B execution once
```

立即停止并封口对应 stage，如果出现：

- hard gate failure；
- truth leakage、future access 或 arm contamination；
- required scenario support 为零；
- pose/keyframe interface 不等价或无法声明 gauge/uncertainty；
- T0 只通过多 abstain 获得表面改善；
- false reacquisition 增加；
- stale anchor 被继续用于方向性引导。

可能终态包括：

```text
W1_T0_WORLD_REFERENT_SIGNAL_ESTABLISHED
W1_T0_HONESTY_GAIN_ONLY_BY_ABSTENTION
W1_T0_NOT_SUPPORTED
W1_T0_NOT_EVALUABLE_DATA_OR_INTERFACE

W1_T1_INCREMENTAL_TRANSLATION_SIGNAL_ESTABLISHED
W1_T1_NO_INCREMENTAL_VALUE
W1_T1_NOT_SUPPORTED
W1_T1_NOT_EVALUABLE_DATA_OR_INTERFACE
```

任何正终态仍只属于 bounded Development/Exploration evidence，不授权长期 world memory、完整 VIO/SLAM、object map、Brain product FSM、Android/default-App 或安全主张。

## 10. 当前唯一 successor

`P1_W1_STAGE_A_OUTCOME_BLIND_IMPLEMENTATION_AND_DATA_SELECTION`，状态为 `NOT_AUTHORIZED / NO_EXECUTION`。

该 successor 只能选择 C0/T0 的最小实现、数据 support roster、deadline/tolerance 和资源预算；不得运行 performance、读取 evaluator outcome、选择 T1 pose source、创建 SLAM 大工程或接 App。
