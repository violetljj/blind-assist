# Segmentation Failure Atlas 与 residual 可标注性 R0 结果

状态：`PILOT_COMPLETE / GATING_INSUFFICIENT / RESIDUAL_WEAKLY_LABELABLE /
TARGETED_EXPANSION_WARRANTED / DEVELOPMENT_MECHANISM_DIAGNOSTIC_ONLY /
FINAL_CONFIRMATION_NOT_ACTIVATED`

日期：2026-08-01（Asia/Hong_Kong）

协议：`DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0`

## 先给结论

新主线的 200-frame pilot 已落地并完成。当前 segmentation false activation 不是一个
“再调面积阈值”就能解决的单机制问题；空间、因果时序和单一高置信门虽然都能降低 false
positive，但召回保留均未达到预声明的最低门，因此二维结论是：

- gating：`INSUFFICIENT`；
- residual labelability：`WEAKLY_LABELABLE`。

错误机制在 4 个 session 上均有跨 session 信号，满足定向扩展的信息增益规则；但只建议
扩到 6 个既有 dev/consumed session，不运行全量 920-frame 扫描。本轮没有执行扩展推理，
也没有选择或上线任何 gate。

## 问题、证据与口径

本轮只读复用 R1 已消费的 200-frame rehearsal、5,043 个 segmentation component、
对应 canonical pixel truth 与冻结 YOLO trace。4 个输入 session 均保持
`r1_consumed_fresh` 的永久降级身份；本结果只是 `DEVELOPMENT_STANDARD`。

组件级 false activation 固定为“候选类别未与同类别 pixel residual truth 相交”。同时
单独保留任意 hazard 相交、YOLO-overlapped hazard、dominant truth class、空间带、
前后观测 IoU、track persistence、置信度和 margin，避免把类别混淆、YOLO 归因歧义与
纯背景误激活压成同一种错误。机制 tag 是非互斥诊断量，面积占比不能相加。

没有 instance correspondence、depth 或 pose。`A_EFFECTIVELY_COVERED` 因而保持
`NOT_EVALUABLE_NO_INSTANCE_CORRESPONDENCE`；YOLO box 与语义 hazard 的重叠只能记为
`ATTRIBUTION_UNCERTAIN`。

## Atlas 结果

5,043 个候选组件中，3,062 个为同类 residual truth false activation；这些组件的总面积为
364,405 pixels。主要非互斥机制如下：

| 机制 | 组件数 | false component area 占比 | 覆盖 session |
|---|---:|---:|---:|
| `UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY` | 1,269 | 53.81% | 4/4 |
| `YOLO_ATTRIBUTION_AMBIGUITY` | 479 | 37.58% | 4/4 |
| `TEMPORAL_FLICKER` | 2,411 | 38.63% | 4/4 |
| `STABLE_HIGH_CONFIDENCE_ERROR` | 94 | 21.23% | 4/4 |
| `SMALL_FRAGMENT_NOISE` | 2,163 | 10.41% | 4/4 |
| `LARGE_WALKABLE_CONFUSION` | 32 | 9.97% | 4/4 |
| `BOUNDARY_DILATION` | 151 | 0.27% | 4/4 |
| `OTHER_FALSE_ACTIVATION` | 122 | 8.72% | 4/4 |

`UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY` 只表示上部视场的背景型激活，不等同于“远处”；
缺少 depth 时，`DISTANT_BACKGROUND_ACTIVATION` 仍为 `NOT_EVALUABLE_NO_DEPTH`。
`TEXTURE_OR_SHADOW_CONFUSION` 也因缺少 appearance label 保持 `NOT_EVALUABLE`。

## 有限 gating probes

baseline B 对 pixel residual 的 aggregate recall 为 `0.309922`，false-positive pixels
为 `1,313,025`；组件 precision/recall 为 `0.452342/0.669643`。本轮只运行预声明的独立
probe，没有做空间 × 时序 × 置信度笛卡尔积。

| Probe | FP reduction | Recall retention | 最低 session recall retention | 判定 |
|---|---:|---:|---:|---|
| lower field | 67.4% | 38.9% | 8.5% | `INSUFFICIENT` |
| central body corridor | 70.8% | 33.8% | 25.6% | `INSUFFICIENT` |
| upper/head band | 69.6% | 24.4% | 12.1% | `INSUFFICIENT` |
| causal 2-of-3 | 36.0% | 72.2% | 71.5% | `INSUFFICIENT` |
| causal 3 consecutive | 69.3% | 38.3% | 36.4% | `INSUFFICIENT` |
| component median confidence ≥ 0.65 | 47.3% | 60.4% | 38.0% | `INSUFFICIENT` |

最接近的 causal 2-of-3 仍只有 `0.722` recall retention，低于 partial 所需 `0.75`；
更不满足 sufficient 所需的 `0.90` overall 与 `0.80` minimum-session retention。
因此不能从同一数据中选一个“最佳门”接入融合或 App。

## Residual 可标注性

200 帧共有 3,158,486 个 canonical hazard pixels：

- pixel residual `canonical_hazard AND NOT frozen_yolo_box_union`：
  2,501,167 pixels，占 79.19%，定义可复算；
- YOLO box 与 canonical hazard 重叠：657,319 pixels，占 20.81%，只能记为
  `ATTRIBUTION_UNCERTAIN`；
- `A_EFFECTIVELY_COVERED`：没有 instance correspondence，无法给出合法像素数。

因此 pixel proxy 自身为 `LABELABLE`，但三态 residual attribution 整体只能是
`WEAKLY_LABELABLE`。它可用于诊断和训练任务设计，不能作为“YOLO 已有效覆盖/未覆盖真实
实例”的确认真值。

## 扩展决定

预声明规则要求某个可行动机制的 false area share 至少 10%，并跨至少两个 session。
`SMALL_FRAGMENT_NOISE`、`YOLO_ATTRIBUTION_AMBIGUITY`、`TEMPORAL_FLICKER`、
`STABLE_HIGH_CONFIDENCE_ERROR` 与 `UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY` 均通过；
故结果为 `TARGETED_EXPANSION_WARRANTED`。

canonical view 中当前只有 6 个合规的非本 pilot session，全部进入候选清单：

- dev：`center_obstacle`、`parallel_boundary`、`step_curb`、
  `lateral_pedestrian_or_ebike` 各 50 帧；
- consumed old blind：`center_obstacle`、`step_curb` 各 60 帧。

该清单只是下一轮输入选择，没有执行 segmentation inference，也没有生成扩展结论。
下一轮只需验证上述跨 session 机制排序和 gating 失败是否复现；若不复现则停止，不再为
追求正结果扩数据或组合门。

## 权限边界与终态

- Atlas：`PILOT_COMPLETE`；
- gating：`INSUFFICIENT`；
- residual：`WEAKLY_LABELABLE`；
- expansion：`TARGETED_EXPANSION_WARRANTED`，尚未执行；
- Confirmation：`NOT_ACTIVATED`；
- product/safety：`NOT_EVALUABLE`；
- Android、QNN/A568、risk、feedback、TTS、振动和默认 App：均未修改、未授权。

机器可复算输出位于忽略目录
`artifacts.local/evidence/dual-loop-segmentation-failure-atlas-r0/pilot-200/`。
