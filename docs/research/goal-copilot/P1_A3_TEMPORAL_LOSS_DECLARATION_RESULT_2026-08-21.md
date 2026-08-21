# P1-A3 temporal loss-declaration result — 2026-08-21

终态：`TEMPORAL_POLICY_INSUFFICIENT / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`

Claim ceiling：`CONSUMED_ADT_DENSE_IDENTITY_TEMPORAL_POLICY_DEVELOPMENT_ONLY`

## 执行完整性

本轮严格按 [`P1-A3 protocol`](P1_A3_TEMPORAL_LOSS_DECLARATION_PROTOCOL_V1.md) 复用 P1-A2 的 frozen
DINOv2-S fixed-reference raw dense correspondence trace。A2 winner threshold 没有继承；四个 raw feature 只做
outcome-blind empirical percentile normalization，再取 median 作为每帧 continuous evidence。

在打开 private truth 前已一次性封存全部 40 个 outputs：

```text
consecutive-frame hysteresis        16
sliding-window vote                  8
leaky evidence accumulator          16
total                               40
second-round sweep                   0
```

ADT cohort、1,296 个 candidate IDs/bboxes/sources、encoder、initial memory、evaluator 与 truth firewall 均未改变；
online memory update、GT reset、global search、new data、Sky、CoTracker/TAPIR 和 learned temporal model 均为 0。

## 冻结 gate 结果

40 个 policy 的 gate accounting：

| Gate | 通过数量 |
|---|---:|
| correct `>=79` | 40/40 |
| wrong `<=488` | 0/40 |
| max wrong-lock `<=3,399 ms` | 20/40 |
| false-loss `<=152` | 0/40 |
| false reacquisition `<=5` | 40/40 |
| TRACKING boundary transitions `<=75` | 40/40 |
| 30-frame reacquisition chatter `<=5` | 40/40 |
| long-loss declarations `>=3/3` | 40/40 |
| all loss-opportunity declarations `>=3/6` | 40/40 |
| evaluator state/event contract | 40/40 |
| 全部 admission gates | 0/40 |

20 个 policy 通过 8/10 gates，另 20 个通过 7/10；没有 policy 通过完整 usability gate，因此终态不是
`TEMPORAL_SMOOTHING_ONLY_DELAYS_FAILURE`，而是 `TEMPORAL_POLICY_INSUFFICIENT`。

预冻结排名下的代表 winner 为 consecutive hysteresis：

```text
low/high percentile threshold    0.30 / 0.70
TRACKING exit run                2 frames
LOST declaration run             8 frames
UNCERTAIN recovery               2 frames
reacquisition confirmation       5 frames
```

| 指标 | P1-A2 | A3 representative | 冻结 gate |
|---|---:|---:|---:|
| correct assertions | 80 | 81 | PASS `>=79` |
| wrong assertions | 445 | 685 | FAIL `>488` |
| max wrong-lock | 2,700 ms | 2,899 ms | PASS `<=3,399` |
| false-loss | 304 | 205 | FAIL `>152` |
| false reacquisition | 29 | 0 | PASS `<=5` |
| TRACKING boundary transitions | 151 | 10 | PASS `<=75` |
| 30-frame reacquisition chatter | 27 | 0 | PASS `<=5` |

Representative 的 685 wrong 中 625 是 background、60 是 other instance；identity switches 为 28。
False-loss 只下降 `32.57%`，未达到预冻结 50%；wrong 相对 A2 反而增加 `53.93%`。

## 不可兼得不是单个 operator 的偶然失败

三类 operator 都出现相同边界：consecutive / sliding / leaky 的最低 wrong 分别为 `601 / 607 / 669`，均高于
488；最低 false-loss 分别为 `205 / 208 / 206`，均高于 152。

全局 minimum-wrong policy 为 `wrong=601 / false-loss=218 / correct=81 / max-lock=2,899 ms`；全局
minimum-false-loss policy 就是代表 winner，`false-loss=205 / wrong=685`。不存在一个接近同时过两门的 frozen
candidate，不允许事后增加 threshold、窗口、operator 或重新聚合 raw evidence。

False reacquisition 与 chatter 的 0 也不能解释为恢复成功：representative 的 `REACQUIRED` event 为 0，
reacquisition precision 为 null、recall 为 `0/6`。Long-target-absence 虽维持 `3/3` loss declaration，但 latency
为 `68 / 61 / 224` frames；temporal smoothing 没有建立及时、可恢复的 persistence state core。

## 路线结论

P1-A2 的 frame-wise dense identity discrimination signal 作为历史 Development evidence继续成立；A3 关闭的是
“在同一 raw fixed-reference DINO evidence 上用简单 temporal policy 即可得到可靠 state inference”这一假设。
继续调 hysteresis/window/accumulator 只会成为 post-outcome rescue。

按预冻结第三终态，唯一 successor 是：

```text
P1_A4_MATERIALLY_STRONGER_TEMPORAL_CORRESPONDENCE_REPRESENTATION_DESIGN
```

只允许设计 CoTracker/TAPIR class 的 stronger temporal correspondence representation 如何在不改 cohort、evaluator
与 claim ceiling 的前提下进入下一轮；本轮不选择模型、不实现、不下载权重、不执行。仍禁止 Sky、A3 threshold
续扫、直接进入 global target re-search、Android/default-App 或产品/safety promotion。

## 本地 evidence identity

Ignored root：`artifacts.local/evidence/p1_a3_temporal_loss_v1/`

```text
temporal_candidates.json 7FF451F26D8BDE67D273E87AEB66C9CA8B7BEA28C1892AF193F0C3A65FBC18B6
winner_prediction.json    68A6DFE075FBF00AC07EF4393F0F6B75D3E1EAF98805745A33C6B714EB9C517B
winner_evaluation.json    85AE6A036F89DBC86383BC9F65E0723C6A0C5A9075759EF40B347272BA469099
sweep_result.json         A58C3E8EF9543F3989F06780C1BA5B803C07F1B5ED61577CAE9448637C002276
```
