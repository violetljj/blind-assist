# P1-A3 temporal loss-declaration and conservative recovery protocol V1

状态：`CONSUMED / TEMPORAL_POLICY_INSUFFICIENT / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`。

结果：[`P1-A3 result`](P1_A3_TEMPORAL_LOSS_DECLARATION_RESULT_2026-08-21.md)。

## 唯一研究问题

P1-A2 已建立 fixed-reference dense identity representation signal，但 frame-wise gate 产生 false loss、false
reacquisition 和 state chatter。A3 不再寻找视觉特征，而只回答：能否把同一连续 dense identity evidence 变成有
记忆、能失信、且不会因单帧反弹而复活的 temporal belief state，同时保住 A2 的主要 wrong/wrong-lock gain。

A2 winner threshold 不继承。冻结继承的只有：

```text
ADT cohort / candidate IDs / bboxes / sources
facebook/dinov2-small@ed25f3a backbone
fixed frame-0 target memory and raw dense correspondence features
frozen P1 evaluator and private truth firewall
```

继续禁止 online memory update、GT reset、global search、new data、Sky、CoTracker/TAPIR、训练和 Android。

## A3 evidence 与运行状态

四个 A2 raw consensus features：`anchor_match_fraction / match_confidence / spatial_consistency /
anchor_coverage`，各自在 1,296 个 consumed RGB candidates 上做无标签 empirical percentile normalization；每帧
evidence 固定为四个 percentile ranks 的 median。A2 的四个 discovered q-threshold 均不进入 A3。

A3 内部状态固定为：

```text
TRACKING -> UNCERTAIN -> LOST -> REACQ_PENDING -> TRACKING
```

只有内部 `TRACKING` 可输出 candidate。`UNCERTAIN` 与 `REACQ_PENDING` 映射到冻结 P1 output 的 `UNCERTAIN`，
`LOST` 映射到 `LOST / LOSS_DETECTED`；`REACQ_PENDING` 通过严格确认后才输出 `TRACKING / REACQUIRED`。任一
低 evidence 或 raw candidate 缺失都会使 pending 回到 LOST。A3 不生成 candidate，因此不能承诺真正 long-term
reacquisition；零 true reacquisition 允许，false reacquisition 优先 fail closed。

## 唯一一次 40-policy sweep

只比较三类确定性 temporal operator：

1. `CONSECUTIVE_HYSTERESIS`：16 个组合；low/high percentile threshold `0.30/0.40 × 0.60/0.70`，tracking exit
   run `2/3`，loss run `5/8`，uncertain recovery 2 帧，reacquisition confirmation 5 帧。
2. `SLIDING_WINDOW_VOTE`：8 个组合；window `5/9`，同一 low/high threshold；exit/loss/recovery vote fraction
   固定 `0.60/0.80/0.60`，reacquisition confirmation 5 个连续 positive windows。
3. `LEAKY_EVIDENCE_ACCUMULATOR`：16 个组合；alpha `0.70/0.85`，同一 low/high threshold，loss hold `4/7`，
   reacquisition confirmation 5 帧。

共 `16+8+16=40`；无 A2 threshold、二轮 sweep、HMM、classifier、LSTM/Transformer/XGBoost、Sky 或模型臂。
全部 40 个 protocol outputs 必须在打开 private truth 前封存并绑定 runner SHA。

## 冻结 gates

A2 references：correct `80`、wrong `445`、max wrong-lock `2,700 ms`、false-loss `304`、false reacquisition
`29`、TRACKING/non-TRACKING boundary transitions `151`、30-frame reacquisition chatter `27`。

A3 admission 必须同时满足：

```text
correct assertions                         >= 79
wrong assertions                           <= 488
max wrong-lock                             <= 3,399 ms
false-loss frames                          <= 152
false reacquisitions                       <= 5
TRACKING/non-TRACKING boundary transitions <= 75
reacquisition chatter within 30 frames     <= 5
LONG_TARGET_ABSENCE loss declarations      >= 3/3
all 6 reacquisition-opportunity loss declarations >= 3/6
frozen evaluator state/event violations    = 0
```

后两项防止用永远停在 UNCERTAIN 的方式伪造 low false-loss。State chatter 只比较冻结 evaluator output 中
TRACKING 与 non-TRACKING 的 boundary transitions；reacquisition chatter 是每个 `REACQUIRED` 后 30 帧内再次离开
TRACKING 的事件数。

通过者按 false reacquisition、false loss、wrong、max wrong-lock、correct retention、boundary transitions、
reacquisition chatter、canonical candidate ID 依次排序。

## 三种穷尽终态

1. 至少一个 policy 通过全部 gates：`TEMPORAL_LOSS_STATE_SIGNAL_ESTABLISHED / NO_POLICY_ADMISSION /
   NO_SCIENTIFIC_VERDICT`。唯一 successor 才可设计 LOST 后的真实 target re-search/proposal mechanism。
2. 没有 policy 全过，但至少一个保留 correct、通过 usability/chatter/loss-declaration gates，却吐回 wrong 或
   wrong-lock safety gain：`TEMPORAL_SMOOTHING_ONLY_DELAYS_FAILURE / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`。
3. 其余：`TEMPORAL_POLICY_INSUFFICIENT / NO_POLICY_ADMISSION / NO_SCIENTIFIC_VERDICT`；这才授权另立 materially
   stronger temporal correspondence representation（CoTracker/TAPIR class）设计。

任何终态都不保留 discovered policy，不修改默认 App，不建立科学、产品或 safety claim。
