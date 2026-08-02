# HFTF Stage C D36 THOR-MAGNI production track veto event result

日期：2026-08-03（Asia/Hong_Kong）

## 结论

终态：

`D36_THOR_MAGNI_PRODUCTION_TRACK_VETO_EVENT_NOT_EVALUABLE`

这不是算法负结果。production `CausalTrackTristateGeometryProducer` 在完整
THOR-MAGNI event replay 中只产生 2 个 admitted contradict frames，来自 2 个
sessions，低于冻结的 `>=10 anchors / >=5 sessions` opportunity gate。因此本次
实验不能回答严格 selected-target track contradiction 是否改善事件效用。

已观察到的 2 次 feedback suppression 均发生在 baseline 已经触发的 event
window 内，没有改变任何 window terminal。不得据此声称“无效”，也不得在同一
outcome 上搜索 track threshold、history length 或 monotonicity 规则实施 rescue。

具体瓶颈：

`SELECTED_TARGET_STRICT_CONTRADICT_COVERAGE_INADEQUATE_FOR_EVENT_VETO`

## 冻结输入与 source 完整性

- cohort：19 个 THOR-MAGNI sessions，530 个 proximity-eligible anchors
  - positive onset anchors：157
  - negative anchors：373
  - positive events：107
- source：D31 冻结 videos、YOLO11n weights 与 Ultralytics `8.4.102`
- source windows：每个 anchor 以前向当前帧结尾的 7 帧窗口，按最接近 15 Hz 取样
- unique source frames：3,710
- person detections：14,364
- truth-free detector TSV 在 evaluator join event truth 前完成

D31 anchor 复现检查全部为零漂移：

- raw detection count mismatch：`0`
- selected-mask mismatch：`0`
- maximum selected-box absolute error：`0.0`

production kernel replay 的 baseline 使用 `OFF`；candidate 只启用
`ACTIVE_CONTRADICT_ONLY` 并注入 production track producer 输出。raw/stable risk
逐帧必须一致，唯一允许差异是 feedback suppression。

## pooled paired outcomes

| 指标 | baseline | candidate | 差异 |
|---|---:|---:|---:|
| positive anchor alerts | 114/157 (72.611%) | 114/157 (72.611%) | 0 |
| positive event hits | 79/107 (73.832%) | 79/107 (73.832%) | 0 |
| negative anchor alerts | 251/373 (67.292%) | 251/373 (67.292%) | 0 |
| candidate-only triggered windows | — | 0 | 0 |
| positive event losses | — | 0 | 0 |

diagnostic counts：

- admitted contradict frames：`2`
- admitted contradict sessions：`2`
- feedback suppressions：`2`
- admitted confirm frames：`28`

## evaluability gates

| Gate | 要求 | 结果 |
|---|---|---|
| complete cohort | 530 anchors / 19 sessions | PASS |
| source parity | count/mask/box 全零漂移 | PASS |
| kernel risk path parity | raw/stable risk mismatch = 0 | PASS |
| baseline event opportunities | positive 与 negative 均非零 | PASS |
| contradict opportunity | >=10 anchors 且 >=5 sessions | **FAIL：2 / 2** |

由于 opportunity gate 失败，后续 effect gates 不获得解释权。表中的 pooled
outcomes 只用于定位覆盖瓶颈。

## 工程故障与重跑边界

以下故障均发生在 truth/outcome join 前，不烧毁 cohort，也不产生科学终态：

1. 首次 producer 使用错误 Python 环境（Ultralytics `8.4.52`），在视频解码前停止。
2. 顺序全视频 decode 过慢，人工终止，未生成 claim artifact。
3. 首版 seek batching 混合 anchor 与 history frames，导致 D31 selected-box 最大
   误差 `0.2387`；在 TSV 与 truth join 前停止。
4. 修复为 anchor-only batch 先执行、history 独立执行后，D31 parity 回到
   `0 / 0 / 0.0`，随后按冻结协议完成。

## 可复现 artifact

目录：

`artifacts.local/evidence/hftf/stage-c-d36-thor-magni-production-track-veto-event-v0/`

- `detections.tsv`
  - bytes：`2,687,484`
  - SHA-256：`5083db4c86ff20c01d12a47aa9b419d3cd1727b8f8ed1383020b33601ad6f731`
- `producer_receipt.json`
  - bytes：`7,844`
  - SHA-256：`26f2991f4f836c2611224bc9ea1c9066d00b6bf03b600155736fafcbf8ab5ade`
- `kernel_replay.tsv`
  - bytes：`65,322`
  - SHA-256：`9401307d5b4a5bce766a94b54f0890031d733cf44144b70d2aca41748a25f25d`
- `report.json`
  - bytes：`31,990`
  - SHA-256：`a3c7861a4b2a1297c6deae1dc9e3464a30043037f003eb533160bec4115ab5d3`

绑定：

- D12 samples SHA-256：
  `9a099a52...1d64a54`
- D31 boxes SHA-256：
  `ecc30d01...7701f`
- weights SHA-256：
  `0ebbc80d...4ee1`

## 下一步

D33/D34 的 track future mechanism 与 production parity 正结果保持成立，但
D36 表明“严格 selected-target contradiction”在当前 event cohort 中覆盖不足。
下一实验只替换这一个变量：复用完全相同的 detections、timestamps、cohort 与
event gates，调用现有 production
`CausalSceneScaleTristateGeometryProducer`，检验 scene-scale contradiction 是否
提供足够事件机会。该 D37 不调整 track threshold，也不改变主线或默认 App。
