# RCLE low-reference false-trigger R1 result

状态：`IMPLEMENTATION_READY_FOR_CONFIRMATION / VALID`

## 结论

冻结四窗中的低参考误触发主要不是 rotation compensation，也不是 observable
support-manager，而是 **pair-level local-flow expansion 的短时越阈波动**。

只实现并评价了一个针对性修订：
`CAUSAL_THREE_PAIR_CONFIRMATION_R1`。它保持原 `0.01/s` 阈值、旋转补偿、
Sparse LK、局部仿射、support-manager、四窗身份和全部 967 pair 不变；只有当前
pair 与同窗前两 pair 均可评价且 expansion 连续越阈时才触发。弃权、窗边界或任一
pair 不越阈都会立即重置；不读取未来 pair。

旧版到新版的角色聚合结果为：

| 角色 | 旧 trigger coverage | 新 trigger coverage | 变化 |
| --- | ---: | ---: | ---: |
| positive | 0.7427609 | 0.7048822 | 保留 94.90% |
| below-reference | 0.3478261 | 0.0250836 | 相对下降 92.79% |

全部预冻结 development gate 通过，因此这一个修订**值得进入另立的新数据外部
验证设计**。这不是 confirmation；下一数据必须在 outcome 前冻结为未见、all-real、
cross-source 的正/低参考 cohort，当前四窗不得重新包装为外部验证。

## 误差归因

[Attribution R0 合同](RCLE_LOW_REFERENCE_FALSE_TRIGGER_ATTRIBUTION_R0_CONTRACT_2026-07-27.json)
在旧 A4 ledger、两个 TUM below 窗和 source-native geometry 已全部被读取的
development 边界内冻结。唯一反事实保持 RGB、pose、rotation compensation、
Sparse LK 与 local affine 不变，只对每个 pair 重置 `PairState`，强制 baseline-only
路径，从而移除 carry/supplement support-manager 影响。

两个 below 窗共 598 pair，旧触发 208 次；其中 10 次 source-native signed radial
expansion 已达到或超过 `0.01/s`，不计入 geometry-below 误触发。剩余 198 次按
预冻结优先级互斥归因：

| 归因 | pair | 占 198 次 |
| --- | ---: | ---: |
| baseline local-flow threshold crossing | 160 | 80.81% |
| rotation-compensation threshold crossing | 26 | 13.13% |
| support-manager-induced trigger | 12 | 6.06% |
| support-manager enabled evaluability | 0 | 0% |

baseline-only 仍触发 `204/598`；旧版与 baseline 的转换为：
`old true / baseline true = 196`、`true/false = 12`、`false/true = 8`、
`false/false = 382`。尤其 `TUM_RGBD_FR2_RPY@7` 的 support-manager 从未激活，
仍有旧触发 `100/299`。这排除了“support-manager 是主要来源”的解释。

rotation compensation 在 geometry-below 误触发中只解释 `26/198`；其余 160 次
在 baseline raw 与 compensated 两条路径都越阈，因此当前最有支持的解释是
pair-level optical-flow/local-affine 输出的短脉冲，而不是补偿方向整体错误。

Attribution producer 的 598-pair 双进程运行墙钟为 `84.5 s`，OpenCV 每进程单线程。
第一次包装调用因前台 stdout 超时而退出，未创建输出目录或结果；保持同一冻结合同
后使用可续接执行单元完成唯一有产物的运行。独立 validator 重算 598 行 attribution
ledger、两窗与 aggregate，结果 `errors=[] / VALID`。

## 唯一修订

[Temporal confirmation R1 合同](RCLE_LOW_REFERENCE_TEMPORAL_CONFIRMATION_R1_CONTRACT_2026-07-27.json)
在读取新版结果前固定以下单一状态机：

```text
above_t = evaluable_t AND expansion_t > 0.01/s
streak_t = streak_(t-1) + 1  if above_t else 0
revised_trigger_t = streak_t >= 3
```

窗边界重置 `streak=0`，lookahead 为 0。没有运行 2-pair、4-pair、majority、
median filter、其他 threshold 或其他 estimator。

## 同窗旧版—新版对照

| 冻结窗 | 角色 | old triggers | revised triggers | old coverage | revised coverage | 首触发额外延迟 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `desk_changing_1@4065.364250422` | positive | 202 | 187 | 0.7481481 | 0.6925926 | 0.073726 s |
| `japanesealley/Hard/P002@000260` | positive | 73 | 71 | 0.7373737 | 0.7171717 | 0.200000 s |
| `TUM_RGBD_FR2_RPY@2` | below | 108 | 5 | 0.3612040 | 0.0167224 | diagnostic only |
| `TUM_RGBD_FR2_RPY@7` | below | 100 | 10 | 0.3344482 | 0.0334448 | diagnostic only |

成功门均在结果前固定：

| Gate | 门 | 结果 | 判定 |
| --- | ---: | ---: | --- |
| below relative trigger reduction | ≥ 0.30 | 0.9278846 | PASS |
| positive trigger retention | ≥ 0.90 | 0.9490027 | PASS |
| max positive first-trigger delay | ≤ 0.25 s | 0.2000000 s | PASS |
| revised positive coverage > below | true | true | PASS |

candidate count 为 1；pair identity、role、timestamp、evaluable 状态与连续 expansion
逐行保持 967/967 相同。独立 validator 重新执行状态机并复算四窗、角色聚合与所有
gate，结果 `errors=[] / VALID`。

## 证据身份

- Attribution contract SHA-256：`3b3b089d524651772869d5893b0c6531f8f0e29d1f9a6d7a373990e3350dbeae`
- Attribution result SHA-256：`fb0f2d39fa6418da51759b5a0cdb7d01b621eece01c3a9440eeca31ed817942f`
- R1 contract SHA-256：`9806211b4ce1b0585ec2c0fbd08b3d52aa72532b797ac225f6390ea1e2f092dd`
- Revised ledger SHA-256：`74d3aa0dd8e386eab8e5760d3ae321eafb2bd75f660b166df89e11560e9bbf8b`
- R1 result SHA-256：`a75bdb16244033051665a46b0edf9cea6f3f3c94a16869bbbb47b6f9583b72e8`
- R1 independent validation SHA-256：`2b7aab715edc7298333f265067c786222f3859f3b89312182b064ddcbb52300e`

本地产物位于
`artifacts.local/evidence/rcle_low_reference_false_trigger_r1/`，不进入 Git。

## 外部验证决策与边界

决策为：`WORTH_SEEKING_NEW_EXTERNAL_VALIDATION_DATA`。

理由不是新版已经“证明泛化”，而是它在同一 development cohort 上同时取得显著的
below 抑制、positive 保留、有限因果延迟与跨四窗一致方向，已达到继续投入新数据的
最小信息增益标准。

下一阶段若启动，必须：

- 在任何 RGB algorithm outcome 前冻结未见、all-real、cross-source 的正/低参考窗；
- 预注册同一个 R1 状态机、原 `0.01/s` threshold、pair 缺失处理和统计；
- 不再用当前四窗选择确认长度、阈值或其他候选；
- 分别报告每来源/每窗结果，不能只用 pooled aggregate 回救；
- 保持 Android、真人、产品与安全权限关闭。
