# HFTF Stage C D33：JRDB detector-track future-range result

日期：2026-08-03

证据角色：Development / detector-bound short-future mechanism replication

研究主线：不变

默认 App：不变

## 结论

D33 只把 D32 的 annotation box/native identity source 替换为真实 stitched RGB
上的冻结 tiled YOLO11n + ByteTrack；七帧 source rule、`+15 frames` future truth、
deadband 与全部 effect gates 均未改变。结果通过所有可判定 gate 与支持 gate：

`D33_JRDB_DETECTOR_TRACK_FUTURE_RANGE_SUPPORTED`

当前正结果提升为：

`JRDB_DETECTOR_TRACK_SHORT_FUTURE_MECHANISM_SUPPORTED`

这证明 D32 的 same-identity causal trajectory hypothesis 不只在 annotation boxes
上成立；在实际 detector/tracker source measurement 上仍能以高精度预测约一秒后
同一行人的相对接近方向。

## 核心结果

| metric | D33 result | frozen gate |
|---|---:|---:|
| source frames | 480/480 | 480 |
| tracked occurrences | 5,366 | descriptive |
| detector/native current matches | 4,772 | >= 400 |
| match IoU median / P10 | 0.770 / 0.528 | descriptive |
| seven-frame + future opportunities | 3,392 | >= 400 |
| non-abstain evidence | 283 | >= 60 |
| selective evidence coverage | 8.34% | descriptive |
| distinct native identities | 25 | >= 15 |
| pooled precision | 96.82% | >= 85% |
| confirm | 128/133 = 96.24% | >= 80% |
| contradict | 146/150 = 97.33% | >= 80% |
| approach prevalence | 30.54% | baseline |
| confirm lift | +65.70 pp | >= +10 pp |
| contradict lift | +27.88 pp | >= +10 pp |
| seven-frame native-ID purity | 96.47% | diagnostic |
| supporting sequences | 3/4 | >= 3/4 |

两个方向都显著超过对应 source-opportunity prevalence，且 history identity
diagnostic 表明 96.47% 的 evidence 在七帧内全部匹配到当前 native identity。
正结果不是靠 current-frame 偶然匹配或单一方向 class imbalance 得到。

## sequence 分解

| sequence | opportunities | evidence | precision | confirm | contradict | history purity | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| Clark Center | 1,100 | 165 | 96.36% | 72/77 | 87/88 | 98.18% | PASS |
| Gates Basement | 830 | 54 | 100.00% | 44/44 | 10/10 | 100.00% | PASS |
| Meyer Green | 161 | 0 | — | 0/0 | 0/0 | — | insufficient |
| STLC 111 | 1,301 | 64 | 95.31% | 12/12 | 49/52 | 89.06% | PASS |

Meyer Green 有 335 个 detector/native matches 与 161 个可评价 source histories，
但没有轨迹同时满足冻结的严格单调 `±0.2/s` non-abstain rule，因此是 sequence-level
`NOT_EVALUABLE`，不是错误方向的负结果。其他三序列全部通过。

## D32 → D33

| metric | D32 annotation source | D33 detector-track source |
|---|---:|---:|
| evidence rows | 480 | 283 |
| pooled precision | 97.50% | 96.82% |
| confirm precision | 96.76% | 96.24% |
| contradict precision | 98.11% | 97.33% |
| distinct native identities | 25 | 25 |
| supporting sequences | 3/4 | 3/4 |

换成 detector/tracker 后 precision 仅下降 `0.68 pp`，两个方向与 distinct-identity
coverage 基本保持。主要损失是 source opportunity 数量，而不是方向正确性。
因此剩余问题更像 selective coverage/realtime integration，而不是 mechanism failure。

## source acquisition 与执行

packet 记录的 480 个 exact image members 在本地已被清理。D33 没有重新构造
manifest，而是直接使用既有 member/CRC/SHA，从 JRDB 官方 `train_images.zip`
做 HTTP Range 恢复：

- 480/480 JPEG SHA verified；
- network bytes：197,136,580；
- 未下载完整约 22 GB archive；
- image receipt SHA-256：
  `93248c797364bccbf11907327375cb5fda49d1b007a415b9bd1c4594feaaf137`。

source producer：

- YOLO11n weights SHA 与 D29/D31 相同；
- 3760×480 panorama 固定五个 overlapping tiles；
- 8,665 raw detections、5,366 tracked occurrences、165 detector tracks；
- tracks SHA-256：
  `efa249fdfe8114dfeb1da419ffdb359189e3d4e6b1f406fabad04a31a39a0fa1`；
- producer receipt SHA-256：
  `fa91162274222b9fe2254ae675ccb95af3fcdd6dca50ab267d476d74764be318`。

最终 report：

- size：177,702 bytes；
- SHA-256：
  `fa2b403328428bbe596833a670970785964ae197e992b39cc47f878b3013984a`；
- detector tracks 与 report 连续重建一次后 SHA 均完全一致；
- 无 detector threshold、tile、tracker、future horizon 或 gate 搜索。

```text
artifacts.local/evidence/hftf/
  stage-c-d33-jrdb-detector-track-future-range-v0/
    tracks.jsonl
    producer_receipt.json
    producer_receipt.json.sha256
    report.json
    report.json.sha256
```

## 证据边界

D33 的正结果层级高于 D32，但仍有边界：

- future native identity/range truth 仍为 JRDB annotation-derived；
- current-frame 评价 association 使用 annotation IoU；
- offline tiled YOLO + ByteTrack 不是 Android realtime 实现；
- 四个短序列、25 identities 仍是 Development evidence；
- 未评价 obstacle event utility、提醒行为、默认 App、产品效果或 human safety。

这些边界限制下一步主张，不把 detector-bound 正结果降写成失败。

## 下一科学变量

D34 不再换模型或 target，而是做 Android shadow state estimator canary：

1. 复用当前 App 的 person detections 与 causal track state；
2. 输出 `CONFIRM_APPROACH / CONTRADICT_APPROACH / ABSTAIN` shadow state；
3. 不改变现有 alert、UI 或默认 route；
4. 在既有 JRDB detector-track replay 上核对 Kotlin/Python decision parity；
5. 在物理设备测量 state latency、track continuity 与输出 census；
6. 只要 parity/runtime 完整即可建立实现可行性，事件效用另行评价。

D34 通过后，才允许冻结一个不驱动提醒的 device shadow replay；支线仍独立于当前
传统主线，直到后续 event-level utility 明确超过主线。
