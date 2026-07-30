# Dual-loop R1 unseen natural event R0 — rank-2 effect result

## 结论

冻结 active R1 在 Shiraz 未见自然 source 上得到：

`FIRST_UNSEEN_SOURCE_NO_EVENT_LEVEL_EFFECT / DENSITY_SIGNAL_ONLY`

安全 guardrail 全部通过，但 5 个 baseline 误提醒负窗没有一个被完整消除。因此不能
声称误提醒事件下降，active R1 不进入默认生产；按预冻结规则以机制、隔离工程和弱
提醒密度信号收口，不在该来源上调阈值、补 hold/latch 或增加新模块。

## 主要结果

| 指标 | baseline | active R1 | 判定 |
| --- | ---: | ---: | --- |
| 正例绝对召回 | 7/7 | 7/7 | 非劣 |
| baseline-hit retention | — | 7/7 | 通过 |
| 250 ms 内 timely retention | — | 7/7 | 通过 |
| 首次提醒新增时延 | — | 4 项 0 ms；3 项 100 ms | 通过 |
| 负窗 corrected | — | 0/5 baseline-false | 无事件效果 |
| 负窗 retained false | 5 | 5 | 未改善 |
| induced false | 0 | 0 | 通过 |
| 全序列 accepted-feedback rows | 508 | 494 | −14，−2.76% |

六个冻结负窗的 pairing 为 `corrected=0 / retained_false=5 /
induced_false=0 / both_clear=1`。5 个 baseline-false 负窗内的反馈行数在 candidate
中逐窗完全相同；全序列少 14 行只能作为重复提醒密度下降，不能代替事件效果。

## 执行与不变量

- 设备：Samsung SM-S9280 / SM8650 / Android 16；
- baseline：4,891 帧 strict QNN HTP，98.026 s，约 49.9 FPS；
- candidate：重放 baseline 的同一 detections/metrics，5.217 s；
- raw/stable risk mutation：0；
- second-loop event mutation permission：0；
- scene-scale contradiction rows：645/4,891；
- candidate trace SHA-256：
  `20a62da4ab412df360703a711a344271b92a1fc24ae15dab4ae4438bc7d767b2`；
- baseline trace SHA-256：
  `b9b1b55890e08fd268cb7d650954651a923fc75c9d537bb1e24721deb5753e9b`；
- effect result SHA-256：
  `6626a4757ad4d9ebe16c316397c364f855193de26adedcc30dc5e3e1205793a6`；
- terminal receipt SHA-256：
  `6b313f015aeb0031338da7878608edd63f4a9555b5f71db6dcd8723a7f7d00cd`。

baseline 与 candidate 使用同一 app APK
`c0de3b13...9a04b` 和 test APK `5d89f4f8...228fe`。candidate 运行前，host 从
truth/input/baseline/build identity 重算出 7 个正例命中和 5 个 alerted 负窗，再将
hash-bound authorization 推送到设备。

## 次要诊断字段说明

candidate producer receipt 的历史字段 `vetoed_feedback_opportunity_count=633`
实际上统计 `candidate_feedback_reason == DUAL_LOOP_CONTRADICTED` 的行，不是实际
baseline-triggered veto 数。由 immutable candidate trace 重新计算：

- contradiction evidence rows：645；
- `DUAL_LOOP_CONTRADICTED` decision-reason rows：633；
- 同帧 baseline triggered、candidate 未触发的实际 veto rows：89；
- 其中 76 行位于冻结 truth windows 之外；
- 净 accepted-feedback rows 只减少 14，因为合法 veto 会改变后续 cooldown/retry
  资格。

该命名偏差不影响正例、负窗 pairing、时延、risk guardrail 或 terminal；论文和汇报
只报告重新计算的 89 与净减少 14，不把 633 称为用户少收到 633 次提醒。

## 证据边界与后继

这是一个 model-reviewed truth、单一 capture、Development 级 canary。它不是人工
临床标注、独立助行或产品安全证据。当前最诚实的双环落地结论是：

1. 简单框尺度三态机制已有独立机制证据；
2. 选择性反证已在真实 Android/QNN trace 上严格隔离并可运行；
3. active R1 保住正例且轻微降低反馈密度；
4. 它没有消除完整误提醒事件，因此默认保持关闭。

若未来新独立来源再次复现“当前帧 veto 后同一负窗立即重试”，才允许提出一个
单变量 R2；本结果本身不授权 latch、IMU、depth、TTC、运动归因或重新调阈值。
