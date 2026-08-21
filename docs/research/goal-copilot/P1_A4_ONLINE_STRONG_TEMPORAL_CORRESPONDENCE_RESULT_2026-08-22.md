# P1-A4 online strong temporal-correspondence result — 2026-08-22

终态：`STRONG_TEMPORAL_CORRESPONDENCE_NOT_SUFFICIENT / NO_PRODUCT_ADMISSION / NO_SCIENTIFIC_VERDICT`

Claim ceiling：`CONSUMED_ADT_MECHANISM_CAPABILITY_PROBE_ONLY`

## 执行完整性

[`P1-A4 protocol`](P1_A4_ONLINE_STRONG_TEMPORAL_CORRESPONDENCE_PROTOCOL_V1.md) 在 implementation selection
之前以 commit `611aa2dc` 冻结；[`selection receipt`](P1_A4_ONLINE_TAPIR_SELECTION_2026-08-21.md) 随后只 smoke
第一候选并固定 official Google DeepMind PyTorch Online BootsTAPIR。Formal runner 又以 commit `f6be8e33` 先于
performance execution 推送。没有运行 JAX fallback、CoTracker、TAPNext/TAPNext++、Cutie 或第二个 checkpoint。

Outcome-blind tracking pass 逐帧 decode/dispatch 既有 P1-D0 15 episodes / 1,724 frames；每个 output frame `t`
只读取 episode-relative `0..t`。完成并封存 prediction 后才执行一次 private evaluation：

```text
oracle frame-0 initializations                  15
post-initialization GT reads                     0
future-frame / GT-reset / semantic reads         0
global target searches                           0
online query-feature replacements                0
frames                                        1,724
fixed point rows                             43,100
evaluator state/event violations                 0 / 0
wall time                                    121.667 s
peak CUDA allocation                     325,331,456 bytes
```

因此本轮是 `EVALUABLE` 的机制性结果，不是 interface、checkpoint、runtime 或 causal failure。

## 冻结 gate 结果

| 指标 | A2 Development reference | P1-A4 | 冻结 A4 gate |
|---|---:|---:|---:|
| correct assertions | 80 | 85 | FAIL `>=120` |
| all wrong assertions | 445 | 812 | FAIL `<=445` |
| background wrong | 422 | 785 | FAIL `<=422` |
| other-instance wrong | 23 | 27 | descriptive |
| identity switches | 27 | 31 | FAIL `<=27` |
| max wrong-lock | 2,700 ms | 2,966 ms | FAIL `<=2,700 ms` |
| false reacquisitions | 29 | 12 | PASS `<=29` |
| false-loss | 304 | 121 | PASS `<=152` |
| temporary-occlusion recovery | 0/3 | 0/3 | FAIL |
| out-of-view return recovery | 0/3 | 0/3 | FAIL |

Correct coverage 只有 `85/777 = 10.94%`，相对 A2 的 `80/777` 仅多 5 帧，不是冻结要求的至少 +50%。
Model/aggregation 产生 926 个 candidate frames；frozen state machine 实际 assertion 的 `897` 帧中只有 `85`
正确，`812` 错误，wrong assertion rate 为 `90.52%`。错误中 `785/812 = 96.67%` 是 background，27 是其他
ADT instance。

Geometry null 的 798 帧分解为：visibility `<6` 573、inliers `<6` 125、inlier ratio `<0.50` 56、coarse
coverage `<4/9` 44。低 false-loss 并不代表 identity persistence 成功；它主要来自 tracker 在目标消失或漂移后仍能
维持一个几何自洽 point set，使状态机少报 LOST，却把错误 background lock 延长。12 次 evaluator-defined
`REACQUIRED` 全是假恢复，reacquisition precision `0/12`、recall `0/6`。

## 分模式证据

```text
CONTINUOUS_VISIBLE    correct 32/540, wrong 441
DISTRACTOR_PRESENT    correct 19/270, wrong 184
TEMP_OCCLUSION        correct 11/93,  wrong 134, recovery 0/3
OUT_OF_VIEW_RETURN    correct 25/72,  wrong 142, recovery 0/3
LONG_LOSS             correct 17/72,  wrong 95,  recovery 0/3
```

这些 mode 可重叠，不能相加成 aggregate。它们共同排除了“主要只是 distractor crossing”这一窄解释：即使 continuous
visible episodes 也只有 `32/540` correct 并产生 441 wrong。当前 correspondence 能维持点轨迹或 coherent affine
geometry，但没有独立证据保证这些点仍属于 frame-0 physical object。

## 机制结论与边界

本轮否定的是：

> fixed frame-0 point identities + official causal visibility + deterministic affine aggregation，本身足以成为可靠的
> object-level persistence representation。

它没有否定 official Online BootsTAPIR 的一般 point-tracking 能力，也不改写 A2 fixed-reference dense identity
verification 的历史 Development signal。A4 恰好把两层分开：强 temporal correspondence 减少了 loss declaration，
但在没有 object-identity verification 时，把“仍可追踪的点/表面”误当成“仍是原物体”。

按预冻结终态 4，当前 implementation 停止：不调 visibility、RANSAC、point grid、state threshold，不重跑，不换
checkpoint，不启动 JAX/CoTracker/TAPNext/Cutie。没有自动 successor；只有另行冻结的新问题才可研究 temporal
correspondence 与 A2 fixed-reference identity verification 的组合，且本结果不能作为该组合有效的证据。

## 本地 evidence identity

Ignored root：`artifacts.local/evidence/p1_a4_online_tapir_v1/`

```text
prediction.json 9D5F08D81535D1739C46D8AF59AFF1C6DF8B33EFE9A9854C2B55819182D39312
result.json     84F7ACA418E4B24969799FA76C6C5D6980A0B55DD851758267CF771DF4B8A156
```
