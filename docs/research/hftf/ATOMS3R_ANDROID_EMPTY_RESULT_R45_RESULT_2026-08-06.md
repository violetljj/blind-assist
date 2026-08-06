# AtomS3R Android 空结果分配优化 R45 结果（2026-08-06）

## 结论

保留 YOLO 解码器的空结果快速路径。当前 AtomS3R QVGA/Q10 10 秒和 1 分钟
回归中 `detection_count=0`，解码器在没有候选框时直接返回共享的空结果，避免
每帧为可变检测列表和空结果包装重复分配。非空候选、坐标映射、阈值、NMS 和
风险语义未改变。

这不是已确认的稳定毫秒级端到端收益；它的价值是降低空场景下的分配与 GC 长尾
风险。当前主要瓶颈仍是设备端相机 `capture→JPEG ready` 阶段。

## 实现

- `YoloOutputDecoder.parse()` 在候选列表为空时返回共享 `YoloDecodeResult`；
- 有候选框时继续走原有 NMS 路径；
- 未改变模型、QNN HTP、输入几何、置信度阈值或风险逻辑。

## 验证

- `JAVA_HOME=E:\\codex-tools\\jdk-17 .\\gradlew.bat :core:vision:testDebugUnitTest`
  通过；
- `:app:assembleDebug` 和 `:app:assembleDebugAndroidTest` 通过；
- SM-S9280 / Android 16 真机 10 秒和 1 分钟通过；
- 1 分钟：1644 帧，0 error，0 reconnect，0 latest overwrite，19 stale dropped；
- ToF：全部有效；
- capture→risk P50/P95/P99：`53.823 / 64.680 / 70.212 ms`；
- capture→JPEG complete P50/P95/P99：`46.217 / 54.172 / 61.638 ms`；
- JPEG decode P50/P95/P99：`1.392 / 3.482 / 3.964 ms`；
- preprocess P50/P95/P99：`0.271 / 0.725 / 0.920 ms`；
- QNN execute P50/P95/P99：`2.861 / 4.181 / 4.665 ms`；
- postprocess P50/P95/P99：`0.669 / 3.767 / 4.550 ms`；
- `detection_count=0`，PSS `186550 → 184084 KiB`。

相对 R38 的 1 分钟结果 `capture→risk=52.81/62.41/66.25 ms`，本次处于同一
波动范围，因此不宣称端到端分位数改善。

## 边界

这是 development-only 性能证据，不代表检测质量、召回、风险一致率、false-clear、
安全性或物理语音/振动起点。原始证据保存在：

```text
artifacts.local/evidence/atoms3r-empty-result-r45/
artifacts.local/evidence/atoms3r-empty-result-r45-1min/
```
