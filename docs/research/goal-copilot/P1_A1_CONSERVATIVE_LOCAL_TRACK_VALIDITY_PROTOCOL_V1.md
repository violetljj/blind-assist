# P1-A1 conservative local-track validity protocol V1

状态：`CONSUMED / VALIDITY_GAIN_ONLY_BY_ABSTENTION / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`

结果：[`P1-A1 result`](P1_A1_CONSERVATIVE_LOCAL_TRACK_VALIDITY_RESULT_2026-08-21.md)。

## 问题与固定干预位置

本轮只回答：frozen P1-R0 local-flow candidate 在漂到 background 前，其 RGB-only tracking-health evidence 是否
足以让系统主动失信。

```text
frozen RGB local-flow candidate generator
  -> candidate bbox + flow internals
  -> NEW validity gate
       valid   -> 原 candidate
       invalid -> null
  -> frozen P1 state machine
  -> frozen evaluator + private GT
```

Gate 不得生成或移动 bbox、改变 candidate identity、更新 initial anchor、全图搜索、触发 oracle reset、读取 GT，
也不得修改 candidate generator、fixed-template reacquisition、P1 state machine 或 evaluator。数据固定为已消费的
P1-D0 15 episodes / 1,724 frames，不新增一帧。

## RGB-only health representation

只在 frozen generator 已准备输出 `sparse_lk_flow` candidate 时记录：

```text
point_survival_ratio          higher is healthier
fb_error_median_px            lower is healthier
affine_ransac_inlier_ratio    higher is healthier
tracked_point_spatial_coverage higher is healthier
flow_residual_dispersion      lower is healthier
bbox_center_jump              lower is healthier
affine_scale_jump             lower is healthier
initial_anchor_appearance     higher is healthier
```

Forward LK、原 error `<30 px`、median translation、`8%` frame jump rejection、appearance `>=0.55` 与原 P1
candidate evidence 完全不变。额外 backward LK 和 affine/RANSAC 只产生 gate evidence。初始 RGB anchor 固定，
不随 tracker 输出更新。

Instrumentation 使用独立 RGB-only replay；在打开 private truth 前，必须逐 episode 验证 replay 的 candidate ID、
bbox/null、source、P1 state/event 与已封存有效 P1-R0 v2 prediction 一致。任何不一致均为
`NOT_EVALUABLE_INSTRUMENTATION_PARITY`，不得 sweep。

## 一次性 compact search

每个 feature 的 threshold grid 只由其 consumed RGB candidate distribution 的 `10%,20%,...,90%` quantile
生成，不读取标签。搜索空间固定为：

1. 每个单 feature threshold；
2. 任意两个不同 feature threshold 的 AND；
3. 唯一三 feature family：`fb_error_median_px AND affine_ransac_inlier_ratio AND
   tracked_point_spatial_coverage`，使用同一 9×9×9 grid。

无 OR、classifier、ML、连续优化、第二轮 grid、结果后调阈值或 Sky。候选排序固定为：

```text
1. correct-assertion retention hard pass
2. maximize episode-macro wrong-assertion reduction
3. maximize frame-aggregate wrong-assertion reduction
4. maximize max-wrong-lock reduction
5. maximize correct retention
6. fewer predicates, then canonical predicate text
```

## 冻结指标与 hard gate

P1-R0 有 `87` 个 correct assertions、`1,221` 个 wrong assertions、max wrong-lock `8,498 ms / 255 frames`。

```text
correct_assertion_retention
  = accepted baseline-correct assertions / 87

Hard:
  retention >= 0.90
  post-init GT reads = 0
  candidate generator / data / model-call count unchanged

Meaningful mechanism signal:
  episode-macro wrong-assertion reduction >= 0.50
  frame-aggregate wrong-assertion reduction >= 0.50
  max wrong-lock duration reduction >= 0.50
```

同时报告 background wrong、other-instance wrong、每 episode wrong reduction、coverage、false loss 与
reacquisition；不生成加权总分。A1 不可能创造新 correct candidate，因此不以 coverage 增长作为目标。

## 三种穷尽终态

1. 至少一个 gate 同时通过 Hard 与全部 meaningful-signal 门：
   `CONSERVATIVE_LOCAL_VALIDITY_SIGNAL_ESTABLISHED / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`。
2. 没有 gate 同时通过，但至少一个不受 retention 约束的 gate 通过全部三项 mechanism 门：
   `VALIDITY_GAIN_ONLY_BY_ABSTENTION / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`。
3. 其余：
   `LOCAL_FLOW_VALIDITY_NOT_IDENTIFIABLE_FROM_CURRENT_RGB_HEALTH_FEATURES / NO_POLICY_ADMISSION /
   NO_SCIENTIFIC_VERDICT`。

第一种终态的唯一后续问题是 `loss -> conservative reacquisition`，但本轮不实现。第二种立即停止 threshold
搜索并要求 materially different validity representation。第三种才允许另立 stronger tracking representation
任务；三种终态都不授权 ReID、Sky、fresh cohort、Android、产品或 safety claim。
