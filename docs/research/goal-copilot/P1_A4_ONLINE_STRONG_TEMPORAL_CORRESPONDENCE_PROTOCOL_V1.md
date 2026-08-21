# P1-A4 online strong temporal-correspondence capability protocol V1

状态：`FROZEN_BEFORE_IMPLEMENTATION_SELECTION / EXECUTION_NOT_STARTED`。

## 唯一研究问题

在只给 frame-0 oracle target bbox、之后零 GT 的条件下，一个 learned、visibility-aware、strictly causal 的
long-term point-correspondence representation，能否同时带来更多 correct physical-identity tracking、更少 background
drift 和至少一种非零遮挡/返回恢复？

P1-A4 不再判断 sparse-LK bbox 是否可信，而是替换 temporal correspondence representation 本身。A1/A2/A3
threshold、DINO gate、temporal policy winner 均不参与 primary decision。P1-D0 15 episodes / 1,724 frames、private
truth firewall 与 frozen evaluator 保持不变。

## 硬因果与权限边界

对 output frame `t`，formal primary implementation 只能读取 `frame 0..t`。Runner 必须逐帧 decode/dispatch，并为
每个 output 记录 `max_source_frame_read`；任何 `max_source_frame_read > t`、整段 tensor 预载、双向 refinement、
future window 或 suffix-conditioned output 均为 `NOT_EVALUABLE_ONLINE_INTERFACE`，不能改称 offline result。

唯一初始化是 frame-0 oracle bbox。之后必须满足：

```text
post-init GT reads = 0
GT/oracle resets = 0
object_uid / visibility GT reads = 0
future-frame reads = 0
semantic detector / ReID / VLM / P0 regrounding = 0
full-frame global target proposal/search = 0
online target/query-feature replacement = 0
```

模型可以保持其 frozen causal recurrent state；不能用当前 prediction 更新初始 query identity。A4 不并行采新数据，
不启用 EgoTracks fallback、Sky、Android 或默认 App。

## 固定 query 与输入合同

每个 episode 把完整 RGB frame 确定性 resize 到 `256×256`，不 crop search area。原始 frame 与 model coordinates
使用独立 x/y scale 精确映射。

frame-0 bbox 内固定采样 `5×5=25` 个 float query points，横纵 normalized offsets 均为：

```text
0.10, 0.30, 0.50, 0.70, 0.90
```

不得按 texture、后续 survival、GT 或 outcome 选择点；不得加入 bbox 外辅助 grid。每个 query 的 identity 与初始
坐标永久不变。每帧 raw trace 至少包含：

```text
point_id
predicted_xy_model
predicted_xy_source
occlusion_logit
expected_distance_logit
visibility_probability
visible / occluded
```

Visibility boolean 与 probability 必须使用所选官方实现 pin 后的官方 postprocessing；不能调 threshold。

## 最小 object aggregation

只在当前 visible target queries 上，从 fixed frame-0 model coordinates 到当前 coordinates 拟合
`estimateAffinePartial2D + RANSAC`：

```text
RANSAC reprojection threshold = 3.0 model pixels
minimum visible points        = 6/25
minimum inliers               = 6
minimum inlier ratio          = 0.50
initial 3×3 coarse-cell coverage among inliers >= 4/9
partial-affine scale          = [0.25, 4.00]
finite coordinates / positive scale required
```

通过时用 affine 变换 frame-0 bbox 四角，取 axis-aligned bounds、映射回 source coordinates 并 clamp；宽高必须
各 `>=3 px`。Candidate evidence 固定为：

```text
identity_support      = min(median inlier visibility probability, inlier ratio)
identity_contradiction = 1 - identity_support
stability             = inlier ratio
oscillation           = 0
```

不通过则当前 candidate 为 null。该单 candidate stream 进入原 P1-R0 deterministic state machine；不增加 A2 gate
或 A3 temporal operator。A4 trace 可以离线附加 A2 DINO diagnostic，但它不得改变 primary output 或选模。

## Outcome-blind implementation selection

先审官方 source/interface/checkpoint/license，再做本机机械 smoke；不得读取 ADT private truth 或跑多个 performance
arms。候选顺序冻结为：

1. official Google DeepMind sequential causal PyTorch BootsTAPIR；
2. 仅当 1 在任何 formal run 创建前机械失败时，official JAX Online TAPIR；
3. 若二者都失败，终止 `NOT_EVALUABLE_ONLINE_INTERFACE`，不自动切 CoTracker/TAPNext/Cutie。

Selection hard checks：单帧 recurrent API；任意 frame-0 query points；tracks + occlusion + expected-distance/confidence；
official reproducible checkpoint；明确 license；本机 CUDA load、one-frame init/predict、25-point shape/finite canary；
峰值显存不超过本机可用容量。只要第一候选全部通过就永久选择它，不继续 smoke 第二候选。

CoTracker3 official online API 的 overlapping `2×step` window、TAPNext/TAPNext++ 与 Cutie 只进入接口审计说明，
不成为本轮模型臂；任何 offline teacher 也不进入 primary comparison。

Selection receipt 必须冻结 official repository commit、source tree SHA-256 manifest、checkpoint URL/SHA-256、license、
Python/PyTorch/CUDA/GPU identity、exact constructor and postprocessing source hashes，随后 formal run 固定不变。

## 冻结 evaluator gates

A2 Development reference：correct `80/777`、wrong `445`、background wrong `422`、identity switches `27`、
max wrong-lock `2,700 ms`、false reacquisition `29`、false-loss `304/777`、temporary recovery `0/3`、out-of-view
return recovery `0/3`。

Hard evaluability：所有因果/权限 receipt 为 0 violation，25×1,724 point rows 与 frame/candidate/output schemas 完整，
frozen evaluator state/event violations 为 0。通过后才判断：

### Identity safety

```text
wrong asserted frames       <= 445
background wrong            <= 422
identity switches           <= 27
max wrong-lock              <= 2,700 ms
false reacquisition         <= 29
```

### Persistence utility

```text
correct identity coverage numerator >= 120/777
false-loss frames                    <= 152/777
temporary-occlusion recovery >= 1/3
OR out-of-view return recovery >= 1/3
```

`correct>=120` 是相对 A2 至少 +50% 的预冻结 capability threshold；不能用 wrong 降低但 correct 仍约 80 的结果
声称 stronger persistence。仍报告 per-mode、background/other-instance、precision/recall、visibility support、point
survival、geometry rejection、latency、峰值 VRAM 与 wall time，不生成加权总分。

## 穷尽终态

1. Hard evaluability、全部 safety 与全部 utility 同时通过：
   `STRONG_TEMPORAL_CORRESPONDENCE_SIGNAL_ESTABLISHED / NO_PRODUCT_ADMISSION / NO_SCIENTIFIC_VERDICT`。
2. `correct>=120` 且至少一种 recovery 非零，但任一 safety gate 失败：
   `CORRESPONDENCE_COVERAGE_GAIN_WITH_IDENTITY_SAFETY_FAILURE / NO_PRODUCT_ADMISSION / NO_SCIENTIFIC_VERDICT`。
3. 全部 safety 通过但 `correct<120`：
   `STRONG_CORRESPONDENCE_GAIN_ONLY_BY_ABSTENTION / NO_PRODUCT_ADMISSION / NO_SCIENTIFIC_VERDICT`。
4. 通过 Hard evaluability，但既未建立 coverage/recovery gain，也未形成安全 abstention：
   `STRONG_TEMPORAL_CORRESPONDENCE_NOT_SUFFICIENT / NO_PRODUCT_ADMISSION / NO_SCIENTIFIC_VERDICT`。
5. 任一 online/interface/runtime/receipt hard check 失败：
   `NOT_EVALUABLE_ONLINE_INTERFACE / NO_ALGORITHM_VERDICT`。

只有终态 1 才允许下一步设计 `A4 representation + A2 fixed-reference identity verification + stable loss semantics`；
终态 2 的唯一后续是 identity verification 设计；终态 3/4 停止当前实现；终态 5 保留为接口失败，不能伪造算法负结果。
任何终态都不准入产品、科学或 safety claim。
