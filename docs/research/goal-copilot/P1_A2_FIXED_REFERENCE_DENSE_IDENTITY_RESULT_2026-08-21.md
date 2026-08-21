# P1-A2 fixed-reference dense-identity validity result — 2026-08-21

终态：`DENSE_IDENTITY_VALIDITY_SIGNAL_ESTABLISHED / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`

Claim ceiling：`CONSUMED_ADT_RGB_DENSE_IDENTITY_DEVELOPMENT_SIGNAL_ONLY`

## 执行完整性

本轮严格按 [`P1-A2 protocol`](P1_A2_FIXED_REFERENCE_DENSE_IDENTITY_PROTOCOL_V1.md) 只消费既有 P1-D0
15 episodes / 1,724 frames。P1-R0 的 candidate ID、bbox、source 与 sealed prediction 通过绑定的 A1 parity trace
保持不变；没有生成、移动或扩大 bbox，没有 global search、added reacquisition、online memory update 或 post-init GT read。

唯一 encoder 为 `facebook/dinov2-small` revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`。公开 RGB 阶段完成：

```text
episodes                           15
frozen sparse-LK candidates     1,296
encoded crops                   1,311  (15 initial anchors + 1,296 candidates)
online target-memory updates        0
post-initialization GT reads         0
```

每个 episode 的 oracle initial crop 只编码一次为永久 256-patch memory。当前 bbox 只产生 mutual dense matches、
match confidence、partial-affine spatial consensus、anchor coverage、dispersion 与 diagnostic global cosine。
完成并封存 `dense_identity_trace.json` 后才打开 private truth。

一次性 policy family 严格为四个 correspondence-consensus feature 各 5 个无标签分位点的 AND：`5^4=625`。
无单 cosine gate、第二轮 grid、encoder/layer/patch sweep、classifier、Sky 或其他模型。625 个 policy 中：

```text
retention >= 90%                              30
wrong reduction >= 60% AND lock reduction >= 60%  593
全部 admission gates                           4
```

## 冻结 admission 结果

按预冻结排名，唯一 top policy 是：

```text
anchor_match_fraction >= q35 = 0.1640625
AND match_confidence >= q35 = 0.755523741
AND spatial_consistency >= q35 = 0.423076923
AND anchor_coverage >= q50 = 0.875
```

| 指标 | P1-R0 | A1 best retention-admissible | P1-A2 | A2 相对 R0 |
|---|---:|---:|---:|---:|
| correct assertions | 87 | 80 | 80 | retention `91.95%` PASS |
| all wrong assertions | 1,221 | 737 | 445 | reduction `63.55%` PASS |
| background wrong | 1,094 | 675 | 422 | reduction `61.43%` |
| other-instance wrong | 127 | 62 | 23 | reduction `81.89%` |
| episode-macro wrong reduction | — | `44.73%` | `46.28%` | diagnostic only |
| max wrong-lock | 8,498 ms | 7,698 ms | 2,700 ms | reduction `68.23%` PASS |

因此 fixed-reference dense correspondence 同时越过冻结的 `90% / 60% / 60%` 三门，终态机械确定为
`DENSE_IDENTITY_VALIDITY_SIGNAL_ESTABLISHED`。这建立的是 materially different representation 在 consumed
Development 上有用的 signal；它不准入当前 discovered threshold，也不证明泛化、科学、产品或 safety 改善。

## 必须保留的负面边界

Frozen evaluator 的完整输出仍很差，不能只看相对 reduction：

```text
correct identity coverage        80/777 = 10.30%
wrong asserted frames               445
identity switches                    27
false-loss frames                    304/777 = 39.12%
evaluator-defined false reacquisitions 29
temporary-occlusion recovery         0/3
long-loss / return reacquisition     0/6
```

这里的 29 次 false reacquisition 不是新增 search candidate；它们来自 validity gate 把同一 frozen candidate
反复删除/恢复后触发 P1 state-machine 的 evaluator event。它直接说明 loss declaration、temporal stability 与
reacquisition semantics 尚未解决，不能把 frame-wise winner 直接接入 App。

`pre_drift_warning_lead` 也没有建立提前预警：14 个含首次 GT-wrong assertion 的 episode 中，正 lead 为 `0`；
`1` 个同帧 warning、`11` 个晚 `66–2,033 ms`、`2` 个从未产生 warning，12 个非空 lead 的中位数为 `-200 ms`。
因此 A2 的主要收益是缩短已经形成的 wrong lock，不是证明它会在 drift 发生前预警。

## 路线结论

A1 的 optical-flow health representation family 正式停止；A2 说明 frozen initial physical-target memory 加
dense correspondence consensus 是有信息的不同 representation。但当前 winner 只是 consumed Development
discovery，存在晚报、false loss 与 state churn。

唯一 successor 收窄为：

```text
P1_A3_LOSS_DECLARATION_AND_CONSERVATIVE_REACQUISITION_DESIGN
```

只允许先设计如何把 dense identity evidence 变成稳定 loss event，以及 loss 后的 bounded conservative
reacquisition；不得保留本轮 threshold、直接实现/运行 reacquisition、续扫 A2、换 encoder、上 CoTracker/TAPIR、
引入 Sky/fresh cohort/Android 或修改默认 App。新协议冻结前 execution=false。

## 本地 evidence identity

Ignored root：`artifacts.local/evidence/p1_a2_fixed_reference_dense_identity_v1/`

```text
dense_identity_trace.json CF24FD7749E835C3F2C7B203361114793BEEC4774F6D1DF9E89644A922ECE471
winner_prediction.json    CDC39885F6ABB3FC7AFA47CEE029778A586A56D19428DC39347573794E437910
winner_evaluation.json    15B76123ACB6A70D9EED911DA1F9C26D7D4BF69316E4BF1C35A6904B3239211F
sweep_result.json         C9D8884C6D763A636DA507DC2802725F53977913D94DB221CDDB23ABB9383733
```
