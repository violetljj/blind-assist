# BlindAssist Assistive Geometry B1 training protocol Attempt 2

状态：`CURRENT / DUAL_ORIENTATION_PROTOCOL_FROZEN / IMPLEMENTATION_NOT_AUTHORIZED`

本文件是 [Attempt 1](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09.md) 的
current overlay。未改字段继续继承 Attempt 1；以下修订取代其单一 portrait shape 和原
DEVELOPMENT sub-role split。

## Full-FOV dual orientation

| family | model tensor | orientation index | TRAIN frames |
|---|---:|---:|---:|
| portrait | `1×3×608×448` | 1 / 3 | 2,724 |
| landscape | `1×3×448×608` | 0 / 2 | 2,076 |

先用 B0 reader 做 upright rotation，再保持 full FOV 映射到对应 shape，并分别更新 K 的
`sx/sy`。禁止跨 family crop/pad/rotation，禁止同 batch 混 shape；sampler 使用 deterministic
orientation buckets。产品部署与产品决策仍只使用 portrait stratum，landscape 不能救
portrait failure。

## Development role 修订

Attempt 1 calibration 只有 30 个 portrait 帧且没有 portrait-dominant parent。Attempt 2 在
outcome 前按 pose orientation 重新固定 identity split，使 calibration/selection 各有一个
portrait-dominant parent：

- calibration：`41127065 / 42444793 / 42444891 / 47430934`，306 portrait frames；
- selection：`42898438 / 42899869 / 44358604 / 47332413`，259 portrait frames。

这仍然只各有一个 portrait-dominant parent，因此 confidence threshold 的 portrait 结论上限
明确为 `DEVELOPMENT_ONLY_SINGLE_PORTRAIT_DOMINANT_PARENT`。后续 independent confirmation
不能回头调 threshold。

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_TARGET_AND_MODEL_IMPLEMENTATION_LOCK`

只授权 TRAIN target cache、dual-shape forward/backward、orientation bucket、K/flip、loss 和
resume smoke；正式训练及 DEVELOPMENT/CONFIRMATION outcome 继续关闭。
