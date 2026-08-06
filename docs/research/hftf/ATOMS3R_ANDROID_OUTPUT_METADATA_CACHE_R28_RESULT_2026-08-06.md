# AtomS3R Android 输出元数据/FloatBuffer 缓存 R28 结果（2026-08-06）

## 结论

保留为 `AI_REALTIME` 性能候选。该改动只缓存 LiteRT 输出 Tensor 的字节数、元素数、形状、数据类型和复用的 `FloatBuffer` 视图，不改变模型、输入、阈值、NMS 或风险逻辑。

## 实现

文件：`core/vision/src/main/java/com/linnan/blindassist/vision/TfliteYoloDetector.kt`

- 初始化时缓存 output tensor 元数据；
- 每帧继续复用 direct output ByteBuffer；
- 每帧复用同一个 `FloatBuffer` 视图并 rewind；
- 输出解码仍使用同一 `FloatArray` 和同一 `YoloOutputDecoder`。

## 1 分钟真机结果

冻结配置：

- AtomS3R：VGA `640×480` / JPEG Q14 / 自动曝光；
- Android：`decodeSampleSize=2`、`maxFrameAgeMs=65`；
- 手机：SM-S9280 / SM8650 / Android 16；
- QNN：真实 HTP 路径；
- 固件：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`。

结果：

- 1636 帧，0 error，0 reconnect；
- source packets `1664`，latest overwrite `0`，stale dropped `28`（约 1.7%）；
- capture→risk P50/P95/P99：`62.07 / 74.07 / 80.22 ms`；
- preprocess P50/P95：`0.42 / 1.09 ms`；
- detector total P50/P95：`5.65 / 12.97 ms`；
- QNN execute P50/P95：`2.88 / 4.15 ms`；
- 初始/结束 PSS：`276677 / 287340 KiB`，短测增加约 `10.7 MiB`，暂不能判定泄漏，后续短测继续观察。

相对 R20 的 1 分钟结果（`59.27 / 70.80 / 74.43 ms`），R28 为 `62.07 / 74.07 / 80.22 ms`，没有证明端到端收益；由于实现风险低，暂保留在工作树作为开发候选，但不晋升默认，也不把本轮差异归因于该缓存。

## 证据与边界

原始证据：`artifacts.local/evidence/atoms3r-output-cache-r28-1min/`。

该结果是 development-only 性能证据，不是检测准确率、风险一致率、false-clear、安全、温升或物理语音/振动起点证据。设备测试结束后已恢复 SVGA/Q10。
