# P1-A1 conservative local-track validity result — 2026-08-21

终态：`VALIDITY_GAIN_ONLY_BY_ABSTENTION / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`

Claim ceiling：`CONSUMED_ADT_RGB_HEALTH_GATE_DISCOVERY_ONLY`

## 执行完整性

本轮按 [`P1-A1 protocol`](P1_A1_CONSERVATIVE_LOCAL_TRACK_VALIDITY_PROTOCOL_V1.md) 只消费 P1-D0 既有
15 episodes / 1,724 frames，不增加数据、模型、candidate、bbox、reacquisition 或 oracle reset。

RGB-only instrumentation replay 对有效 P1-R0 v2 的 1,296 个 `sparse_lk_flow` candidate 增加八个 health
features；15/15 episodes 的 candidate ID、bbox/null、source、P1 state/event 与 sealed prediction 完全一致，
`post_initialization_gt_reads=0`。只有 parity PASS 后才打开 private truth。

一次性搜索严格为：

```text
single feature gates                       72
two-feature AND gates                   2,268
one frozen three-feature AND family       729
total                                   3,069
second-round grid / OR / classifier / ML    0
```

## 冻结 hard gate

Baseline reference：`correct=87 / wrong=1,221 / max wrong-lock=8,498 ms`。Retention hard gate 为
`>=90%`；实际 3,069 个 gate 中只有 47 个通过。

Primary episode-macro 排名下，最佳 hard-admissible gate 是：

```text
bbox_center_jump <= q90 = 0.130477211
AND
initial_anchor_appearance >= q30 = 0.792827129
```

| 指标 | P1-R0 | best hard-admissible | 变化 |
|---|---:|---:|---:|
| correct assertions | 87 | 80 | retention `91.95%` PASS |
| all wrong assertions | 1,221 | 737 | `39.64%` reduction FAIL `<50%` |
| background wrong | 1,094 | 675 | `38.30%` reduction |
| other-instance wrong | 127 | 62 | `51.18%` reduction |
| episode-macro wrong reduction | — | `44.73%` | FAIL `<50%` |
| max wrong-lock | 8,498 ms | 7,698 ms | `9.41%` reduction FAIL `<50%` |

这不是一个 background-drift-specific mechanism：它对 other-instance 的相对削减反而更高，并且三个持续可见
episode 的 wrong locks 完全不变。所有 hard-admissible gates 中，macro reduction 最大 `44.73%`、aggregate
reduction 最大 `43.57%`、wrong-lock reduction 最大 `65.89%`，但由不同 gate 取得；没有一个 gate 同时通过
三项 meaningful-signal 门。

最佳 single feature 是 `initial_anchor_appearance >= q40`：保留 `81/87=93.10%` correct，wrong 降至 712，
但 episode-macro / aggregate / wrong-lock reduction 只有 `38.84% / 41.69% / 12.94%`。冻结的
`FB + RANSAC + spatial support` 三特征 family 没有一个 gate 达到 90% retention；这批 evidence 不支持把前三项
flow coherence 直接准入为 conservative validity policy。

## 为什么是 abstention terminal

不受 retention 约束的 safety winner 是：

```text
fb_error_median_px <= q10 = 0.004848426 px
AND
tracked_point_spatial_coverage >= q80 = 0.721590909
```

它把 wrong assertions、switches 和 wrong-lock 全部降到 0，但只保留 `15/87 = 17.24%` correct assertions；
冻结 evaluator 的 correct coverage 降至 `15/777 = 1.93%`，false-loss 升至 `732/777 = 94.21%`。保留下来的
15 帧就是每 episode 的 oracle initialization。安全改善完全来自拒绝所有 post-init assertions，因此机械终态为：

```text
VALIDITY_GAIN_ONLY_BY_ABSTENTION
NO_POLICY_ADMISSION
NO_SCIENTIFIC_VERDICT
```

不得保留任何 discovered threshold、继续细扫 quantile、把 0 wrong 宣称为安全改进，或进入 loss/reacquisition。
当前 local-flow health representation 没有在保留 90% baseline-correct assertions 的同时建立足够的提前失信信号。

## 唯一 successor

```text
P1_A2_MATERIALLY_DIFFERENT_TRACK_VALIDITY_REPRESENTATION_DESIGN
```

下一步只允许设计 materially different 的 tracking-validity representation；不能续调 A1 threshold。可讨论更强的
correspondence / tracking representation、非漂移更新的 template memory 或 modern tracker，但仍不先做 ReID，
不改 frozen evaluator/cohort/truth firewall，不运行 Sky、fresh data、Android 或产品/safety promotion。设计协议
冻结前不执行 A2。

## 本地 evidence identity

Ignored root：`artifacts.local/evidence/p1_a1_conservative_local_validity_v1/`

```text
flow_health_trace.json  36C2E2B9E9188A3215521A26B514C6BC4F0D663E1D4CCEF04A66BD0E53A4E7E2
winner_prediction.json  F9CD0FA94338A28E044506AA5ECE02B8C9F06C3C62A7A2E44E6CF2D734E17DA1
winner_evaluation.json  4908C79B9875FCB46138B3A0028A5B84BF081330F689484A748CF6414368834B
sweep_result.json       1E14877FC5A63D01844466851FF4F93ECAED1433DB531D2D9931B0F815EC97AF
```
