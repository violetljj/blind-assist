# BA-ADT-INSTANCE-REDETECTION-1 Development result

状态：`VALID / DEVELOPMENT_ONLY / BOUNDED_LOCALIZATION_UTILITY / WRONG_INSTANCE_ZERO_ON_SEQ136 / REACQUISITION_BOTTLENECK_NOT_RESOLVED / SKY_DISABLED`

## 问题与实现

本轮只问：在冻结 Goal Copilot 之前，RGB observer 能否在 5-frame optical-flow 丢失后，依据已经确认的
目标外观保守地重接同一 instance。输入继续是同一已消费 Development sequence
`Apartment_release_clean_seq136_M1292 / Carrot_A (uid 4917588638317799)`；ADT GT 只进入后置 evaluator。

实现保留原 YOLO11n + flow5 快环，并增加：

- immutable anchor 加最多 5 个 trusted RGB crop templates；
- LOST 时把同类全图候选门从 `0.10` 降至 `0.02`，但低分候选不能直接恢复 visible；
- 以颜色/梯度 crop embedding 为主要身份项（权重 `0.80`），长宽比、尺度和随时间衰减的位置只作弱先验；
- appearance `>=0.82`、组合分 `>=0.75`、top1/top2 margin `>=0.08`；
- 长时重检测必须 3 帧窗口内至少 2 次相容；短时连续性只允许高置信且空间一致的 detector reconnect；
- 重接后 8 帧禁止更新长期记忆；不确定时保持 LOST。

Observer CLI 仍不接受 GT 参数。最终运行显式使用本机 CUDA `device=0`；设备只改变推理执行面，不改变
模型、RGB 输入、阈值或 evaluator。

## Held-forward 结果

两臂都由 evaluator v3 在前 25% 选择固定 offset、后 75%（2,160 帧，其中 GT-visible 1,064 帧）计分。

| Metric | flow5 baseline | instance-redetection R1 |
|---|---:|---:|
| Localization recall, IoU >= 0.10 | 0.5808 | **0.6203** |
| Mean IoU on GT-visible frames | 0.4469 | **0.4743** |
| GT-invisible false-visible | 0.0073 | **0.0073** |
| Longest localization dropout | 162 | **159** |
| Normalized bearing MAE | 0.00648 | **0.00638** |
| Correct reacquisition within 30 frames | 0.4000 | **0.4000** |
| Correct reacquisition within 90 / 180 frames | 0.5000 / 0.5000 | **0.5000 / 0.5000** |
| Median successful reacquisition delay | 28 frames | **26 frames** |
| Instance-redetection events | 0 | 13 |
| Correct / wrong annotated instance / unresolved | n/a | **13 / 0 / 0** |
| ID switches at instance redetection | n/a | **0** |

R1 将 recall 提高 3.95 个百分点、mean IoU 提高 2.74 个百分点，且没有增加该 sequence 上的
false-visible 或 annotated wrong-instance event，因此建立了 bounded Development utility。它没有提高
`@30`、`@90` 或 `@180` 的成功比例，最长 dropout 只缩短 3 帧，所以不能声称长时重捕获问题已解决，
也不构成 M1 tracking completion。

`wrong-instance=0` 只表示这 13 次重检测没有落到 evaluator 可见的其他 ADT instance；本 sequence 没有
第二个同 prototype carrot，不能外推为相似实例干扰下的零错误保证。handcrafted embedding 与 YOLO11n
低阈值同类候选仍是当前明确上限。

## Evidence identity

```text
rgb_observations_final.json  sha256 2b1d9d6d9b3b7548fe98dadd597d5403c56266417c747484daf66480973c249f
evaluation_baseline_v3.json  sha256 42f10c0f8b9600126142e4c840c3e0a25d9b32d0a3a98516971f6a1be03efcd5
evaluation_final.json        sha256 cf05386428d36ddbc096118280dbb1ba04d3e2fbe932e0c6df2eaa236127e26d
```

机器输出位于 ignored `artifacts.local/evidence/ba_adt_instance_redetection_1/`。本轮没有运行 Sky、修改
冻结 Goal Copilot、生成产品/安全结论或改变默认 App。

## 决策与唯一下一步

R1 作为 RGB observer 的 Development candidate 保留，但不替代“长时重捕获仍不足”的终态。唯一
successor 是 `ADT1_LEARNED_INSTANCE_REDETECTION_R2`：保持本轮状态机、确认门、evaluator 和 flow5
不变，只把候选生成/身份表征升级为 bounded `YOLOE visual prompt + DINOv2 ViT-S/14` canary，并首先
检验此前从未成功的 GT-visible segments 是否获得候选。若 learned candidate coverage 仍不能提高
reacquisition success，则停止该 RGB-only 2D instance route，转向显式 identity ambiguity/3D object memory，
而不是降低 wrong-instance 门或调用 Sky。
