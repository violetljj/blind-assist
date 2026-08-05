# AtomS3R-M12 + ToF4M Android 端到端计时 R0

## 结论

`SM-S9280 / Android 16` 上，AtomS3R-M12 的 XGA JPEG 流进入 BlindAssist 当前生产检测与风险链后，五分钟 `capture -> risk complete` 为：

| 阶段 | P50 | P95 | P99 | 最大值 |
| --- | ---: | ---: | ---: | ---: |
| capture -> Android JPEG complete | 102.1 ms | 135.1 ms | 164.6 ms | 1679.9 ms |
| capture -> JPEG decode complete | 128.2 ms | 175.4 ms | 209.9 ms | 1685.8 ms |
| JPEG decode | 3.59 ms | 3.92 ms | 4.53 ms | 55.1 ms |
| Bitmap -> RGBA owned copy | 50.9 ms | 52.9 ms | 55.2 ms | 135.9 ms |
| production detector stage | 47.3 ms | 48.1 ms | 50.6 ms | 150.7 ms |
| capture -> risk complete | **227.4 ms** | **274.4 ms** | **310.6 ms** | **1786.7 ms** |

这是 Development-only 性能基线，不是准确率、安全性、人体试验或产品可用性证据。

## 身份与冻结配置

- 日期：2026-08-06（Asia/Hong_Kong）
- Android 设备：Samsung `SM-S9280`，product `e3qzcx`，SDK 36 / Android 16
- App：`com.linnan.blindassist`，versionCode 37，versionName 10.9.0
- App APK SHA-256：`a3af06fa712fc250a26737574df9e474bb696b95e28d599e34f3e4cebe725a4d`
- Test APK SHA-256：`acc0ec0d65e5a7c57c47b6ea377e56c25e98df7f940fc28fb72c0f822a80534d`
- 固件：`atoms3r_m12_tof4m_stream_r11_per_frame_copy_buffer`
- 设备 endpoint：`http://192.168.5.11`
- 相机：XGA 1024x768，JPEG quality 10，brightness 1，自动曝光开启
- ToF：采样开启；相机—ToF 外参标定仍暂停，ToF 不参与风险融合
- 实际 detector 路由：production provider 先尝试 QNN，初始化失败后明确回退到 `CPU_XNNPACK`；本结果不得称为 QNN/NPU 延迟

## 正式运行

- 项目内 instrumentation：`AtomS3rAndroidE2eTimingTest.recordsOptInRealDeviceEndToEndBaseline`
- 请求时长：300 s
- 测试记录时长：302.497 s
- runner：`OK (1 test)`，外部命令耗时 302.738 s
- 手机处理帧：2890（约 9.55 fps）
- MJPEG reader 读入：6620（约 21.88 fps）
- latest-only 覆盖：3730；`processed + overwritten = packets_read`
- 重连：0
- 流错误：0
- 对时：10 次成功，0 次失败
- ToF valid：2890/2890
- TTS API 请求：0
- vibration API 请求：0

手机处理能力低于设备流输入能力，因此容量 1 的 latest-only 槽位覆盖 3730 帧（56.34%）。这表示系统主动选择最新画面，不存在主机端旧帧排队；但当前生产 CPU 路由只处理约 9.55 fps，后续若要求更高有效感知帧率，优先优化 Bitmap-to-RGBA 复制或检测输入路径，而不是扩大队列。

## 时钟与 ToF

设备 UDP 3333 使用 7 次 midpoint ping 中最小 RTT 样本，每 30 秒刷新：

| 指标 | P50 | P95 | 最大值 |
| --- | ---: | ---: | ---: |
| sync RTT | 6.63 ms | 9.38 ms | 9.38 ms |
| sync error bound | 3.31 ms | 4.69 ms | 4.69 ms |
| ToF age at JPEG ready | 58.3 ms | 104.4 ms | 145.5 ms |

外设 capture 时间已映射到 Android `elapsedRealtimeNanos()`；每帧保留 offset、RTT 和误差界。ToF 样本仍是逐帧绑定诊断元数据，不授权几何融合。

## 抖动与稳定性

处理帧 capture interval 为 P50/P95/P99 `108.0/144.2/180.1 ms`。每分钟的 `capture -> risk complete`：

| 分钟 | 帧数 | P50 | P95 |
| --- | ---: | ---: | ---: |
| 1 | 578 | 222.1 ms | 257.3 ms |
| 2 | 568 | 233.7 ms | 288.4 ms |
| 3 | 581 | 230.3 ms | 274.0 ms |
| 4 | 582 | 222.6 ms | 257.0 ms |
| 5 | 581 | 228.3 ms | 289.1 ms |

存在一个真实的约 1.68 s `capture -> JPEG complete` 停顿，发生在 frame 9541；同一 clock offset 下前后帧恢复到约 100–130 ms，因此不是周期对时切换伪影。五分钟内没有重连或流错误，P99 未被该单帧主导，但该尾帧应保留为后续 Wi-Fi/设备写出诊断对象。

PSS 从 57,057 KB 增至 212,814 KB。起点在首次模型推理前，包含模型/运行时惰性加载，因此不能据此声称 155 MB 内存泄漏，也不能声称内存稳定；需要后续增加 warm-up 后的分钟级 PSS 采样再评价内存趋势。

设备测试结束后 `stream_clients=0`，Wi-Fi RSSI `-37 dBm`，设备内部温度传感器读数 `65.1 C`（不是环境/外壳温度），可用 heap 149,104 bytes，Wi-Fi reconnect attempts 0。

## 反馈边界

当前画面未形成可反馈风险，因此本次没有 TTS 或震动 API 请求：

- `risk -> TTS API request`：`NOT_EVALUABLE_NO_TRIGGER`
- `risk -> vibration API request`：`NOT_EVALUABLE_NO_TRIGGER`
- 实际声音开始：`NOT_EVALUABLE`
- 实际振动开始：`NOT_EVALUABLE`

API 请求时刻即使未来有记录，也不能替代麦克风/加速度计测得的物理起点。

## 本地证据

最终原始 JSONL 位于 ignored `artifacts.local/evidence/atoms3r-android-e2e-5min-r0-final-20260806/`：

- frame rows：`frames-1785950133035.jsonl`
- summary：`frames-1785950133035-summary.json`
- computed metrics：`computed_metrics.json`
- frame JSONL SHA-256：`bc5e383231cdbceb601bc78de5806d5a4c259d92e19141859272bc1243f2ead9`

该本地目录不是 Git 追踪证据仓；复现实验必须重新绑定当前 Git、APK、固件、设备和 instrumentation 输出。
