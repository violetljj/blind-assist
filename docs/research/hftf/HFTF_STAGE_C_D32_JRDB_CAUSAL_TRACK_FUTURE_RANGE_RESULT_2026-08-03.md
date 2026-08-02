# HFTF Stage C D32：JRDB causal track future-range result

日期：2026-08-03

证据角色：Development / native identity-bound short-future mechanism canary

研究主线：不变

默认 App：不变

## 结论

D32 在冻结协议下直接复用四个既有 JRDB native multisensor observation packet，
用同一身份过去七帧的 `log(box_height)` tri-state，预测 frame `+15`
（约一秒）时的相对 range direction。结果通过全部可判定 gate 与全部支持 gate：

`D32_JRDB_CAUSAL_TRACK_FUTURE_RANGE_SUPPORTED`

当前建立的正结果是：

`JRDB_ANNOTATION_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED`

这不是“系统尚未证明所以结果无效”。它在实际证据层明确证明：一个极小、因果、
identity-bound 的轨迹状态，能高精度预测同一行人约一秒后的相对接近方向。

## 核心结果

| metric | result | frozen gate |
|---|---:|---:|
| future opportunities | 8,766 | descriptive |
| non-abstain evidence | 480 | >= 80 |
| evidence coverage | 5.48% | selective by design |
| distinct sequence-bound tracks | 25 | >= 20 |
| pooled precision | 97.50% | >= 85% |
| confirm rows | 216 | >= 20 |
| confirm precision | 96.76% | >= 80% |
| contradict rows | 264 | >= 20 |
| contradict precision | 98.11% | >= 80% |
| approach prevalence | 51.52% | baseline |
| confirm lift over prevalence | +45.24 pp | >= +10 pp |
| contradict lift over prevalence | +49.62 pp | >= +10 pp |
| sequences with >=10 evidence rows | 3/4 | >= 3/4 |
| supporting sequences | 3/4 | >= 3/4 |

该结果不只是由总体 class prevalence 造成：两个方向各自都比对应 opportunity
prevalence 高约 45–50 个百分点。

## sequence 分解

| sequence | evidence | precision | confirm | contradict | gate |
|---|---:|---:|---:|---:|---|
| Clark Center | 233 | 97.00% | 109/116 | 117/117 | PASS |
| Gates Basement | 91 | 100.00% | 72/72 | 19/19 | PASS |
| Meyer Green | 9 | 100.00% | 9/9 | 0/0 | insufficient rows |
| STLC 111 | 147 | 96.60% | 19/19 | 123/128 | PASS |

Meyer Green 只有 9 条 evidence，因此按冻结规则不计入 supporting-sequence gate；
它是可见的正观察，但不被四舍五入成正式 sequence pass。其余三序列全部通过。

## 与前序结果的关系

- D27 已证明 THOR target 中存在强 history-kinematic information ceiling；
- D28/D29 表明 whole-frame RGB distillation 与 low-resolution
  box-within-flow 没能恢复该信息；
- D30/D31 表明单帧 person bearing 可测，但 box-height distance calibration
  跨 source 不稳定；
- D32 进一步定位：真正有力的变量不是单帧绝对距离，而是
  **same-identity causal trajectory state**。

因此 D28/D29 的负结果不应被解释成“历史运动无用”；它们说明错误发生在
representation/measurement binding。D32 给出了一个可工作的最小机制。

## 工程执行

第一次命令从 `scripts/research/hftf` 运行时，在打开任何 packet 前因默认相对路径
错误触发 `FileNotFoundError`。这被分类为 engineering failure：

- 没有读取任何 source payload 或 future outcome；
- 没有写科学终态；
- 没有烧毁 cohort；
- commit `329f700` 仅把默认路径锚定到由 `__file__` 解析出的 repo root；
- 五个单元测试通过后，在原冻结估计量与 gate 下重跑。

这次故障及其修复是“治理服务科学”的直接实例，而不是协议作废理由。

最终报告：

- report size：226,446 bytes；
- report SHA-256：
  `81761e24b2098d9f585d8c8fd9a786eea0e21fff22c9b99b55dfb017dd07c2ec`；
- 四个 packet 的 SHA-256 均随报告记录；
- 删除非科学的 wall-clock 字段后连续两次重建得到相同 report SHA；
- 无训练、无 threshold/horizon/deadband 搜索、无结果后换 gate。

```text
artifacts.local/evidence/hftf/
  stage-c-d32-jrdb-causal-track-future-range-v0/
    report.json
    report.json.sha256
```

## 证据边界

该正结果仍有明确边界：

- 2D box 与 identity 来自 JRDB annotation，不是 live detector/tracker；
- 3D centers 为 annotation-derived 且标记为 interpolated；
- 480 rows 是 25 条 track 内的纵向重复观察；
- 只有四个短序列，属于 Development mechanism evidence；
- 未评价 event utility、Android runtime、默认 App、产品效果或 human safety。

这些边界限制结论的层级，不否定已经成立的短未来预测机制。

## 下一科学变量

D33 只替换 source measurement，不改变 D32 的 future estimand：

1. 在 JRDB 原始 stitched RGB 上运行冻结 person detector + causal tracker；
2. source 决策只读 detector/tracker 产生的 identity 与 boxes；
3. evaluation 时才把 source tracks 与 native 3D identities 匹配；
4. 复用七帧、`0.2/s`、`+15 frames`、`0.1 m/s` 与 direction gates；
5. 报告 association coverage、identity switches、direction precision 与
   prevalence lift；
6. 工程失败允许修复重跑，结果不足为 `NOT_EVALUABLE`，可判定失败才是科学负结果。

D33 若支持，就建立 detector-bound forward state mechanism，足以进入 Android
shadow state estimator canary；若不支持，则明确瓶颈是 detection/association，
而不是 future-motion hypothesis。
