# 手机设备准入预检 P0A R0

任务 ID：`HETEROGENEOUS_PLATFORM_P0A_R0`（评估对象：当前连接手机）

日期：2026-07-31（Asia/Hong_Kong）

## 结论

当前评估对象不是 A568，而是可持续通过 ADB 访问的 Samsung `SM-S9280` 手机，SoC
标识为 `SM8650`。该手机在以下明确范围内达到：

```text
PLATFORM_ADMITTED_FOR_CANARY
```

准入范围仅包括 LiteRT/TFLite CPU（4 threads）和 LiteRT GPU delegate。QNN HTP/NPU
子路径不准入，终点为 `HOLD_NOT_EVALUABLE`：能力探测报告支持，但实际 HTP delegate
初始化时设备端缺少 `libQnnHtpV75Skel.so`，生产路由因此回退到 CPU。这个 HOLD 不阻止
CPU/GPU canary，也不构成 NPU 性能结论。

这次 P0A 只证明手机可评估、固定帧输入输出链路可复算，以及 CPU/GPU 路径在短时运行中
可持续；不证明跨平台公平性能、不证明功耗效率、不证明产品或安全有效性。T4 需要第二个
固定平台，T5 仍未启动。

## 1. 预检门禁

| 预检项 | 状态 | 证据 |
| --- | --- | --- |
| 准确板卡、SoC、CPU/GPU 与内存 | `PASS` | `SM-S9280 / SM8650 / arm64-v8a`；`MemTotal=11,350,704 kB`；板级标识 `pineapple`；GPU hint 为 Adreno |
| SDK、推理运行时与版本 | `PASS`（CPU/GPU） | Android 16 / API 36；LiteRT 1.4.2；QNN 2.47.0 已打包并可探测 |
| 设备真实可持续访问 | `PASS` | ADB health check 为 `ready`；serial `R5CX10M8Y8X`；连续 instrumentation 与 soak 均完成 |
| YOLO11n 原始或等价导出模型加载 | `PASS`（CPU/GPU） | `yolo11n_fp16_320.tflite`，输入 `1x320x320x3`，输出 `1x84x2100`；模型 SHA-256 见下文 |
| 固定帧逐帧结果、时延、错误导出 | `PASS` | 10 张固定评测图；30 次 App detector run；失败数为 0；已导出 JSON、Markdown、CSV 与错误清单 |
| 温度与系统日志能力 | `PASS` | Thermal HAL AIDL 3 connected，thermal status `0`；ADB/logcat 可复算 |
| 功耗/能耗 | `NOT_EVALUATED` | 运行期间设备通过 USB 连接；没有外部功率仪表，电量和温度不替代能耗测量 |
| QNN HTP/NPU 实际初始化 | `HOLD_NOT_EVALUABLE` | HTP capability `is_supported=1`，但 device-side skeleton 缺失，delegate apply 失败；生产路由回退 CPU |

## 2. 设备与运行时身份

| 属性 | 实测值 |
| --- | --- |
| ADB serial | `R5CX10M8Y8X` |
| manufacturer / model | `samsung / SM-S9280` |
| product / device | `e3qzcx / e3q` |
| board / platform | `pineapple / pineapple` |
| SoC | `SM8650`，manufacturer `QTI` |
| Android | `16`，API `36` |
| ABI | `arm64-v8a` |
| GPU runtime hint | `ro.hardware.egl=adreno`；driver `com.qualcomm.qti.gpu.drivers.pineapple.api34` |
| online CPU | `0-7`（8 个在线逻辑 CPU） |
| CPU max frequency | CPU0-1 `2.2656 GHz`；CPU2-4 `3.1488 GHz`；CPU5-6 `2.9568 GHz`；CPU7 `3.3984 GHz` |
| MemTotal | `11,350,704 kB` |
| LiteRT/TFLite | `1.4.2`；CPU 路径使用 4 threads；GPU delegate 可初始化 |
| Qualcomm QNN | `2.47.0`；HTP FP16 capability probe 可执行，但实际 delegate 初始化失败 |
| 模型 | `yolo11n_fp16_320.tflite` |
| 模型 SHA-256 | `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2` |

本报告不把 CPU implementer/part 数值映射为未经设备清单确认的商业核心名称，也不把
`SM8650` 手机身份解释为 A568 身份。

## 3. 固定帧准入运行

为了在没有候选模型的情况下先验证手机链路，本次使用窄范围 `BaselineOnly` 模式，只
运行现有默认 `yolo11n`，没有改变默认模型或生产路由。评测集为 `BlindAssist EvalSet`，
固定使用前 10 张图 `blindassist_eval_000001.jpg` 至 `blindassist_eval_000010.jpg`；
每图运行 3 次，纯 Interpreter 预热 3 次、测量 10 次。

| 指标 | 结果 |
| --- | ---: |
| CPU backend | `tensorflow-lite-cpu-4threads` |
| Interpreter init | `50.181 ms` |
| 纯 Interpreter invoke P50 / P95 | `42.348 / 42.661 ms` |
| App detector runs / failures | `30 / 0` |
| preprocess P50 / P95 | `9 / 10 ms` |
| inference P50 / P95 | `42 / 42 ms` |
| postprocess P50 / P95 | `1 / 1 ms` |
| App detector total P50 / P95 | `53 / 53 ms` |
| 10 图 diagnostic AP50 / recall / F1 | `0.253 / 0.258 / 0.400` |

最后一行只是这 10 张图上的诊断统计，不是手机准入门槛，也不是安全或产品质量结论。
逐帧输出、时延、错误和资产清单已写入本地证据目录；没有把原始帧或大型 payload 提交到
仓库。

## 4. CPU/GPU 短时稳定性

固定同一评测资产循环、目标 10 FPS，分别运行约 60 秒。生产路由一行的实际 backend
是 CPU fallback，不应被解读为 QNN 成功。

| 路径 | 实际时长 | 帧数 / 有效 FPS | 初始化 | detector total P50 / P95 | inference P50 / P95 | failures | 电池温度 | thermal status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `PRODUCTION_ROUTE -> CPU_XNNPACK` | `60.045 s` | `587 / 9.776` | `96.82 ms` | `54 / 57 ms` | `42 / 43 ms` | `0` | `30.2 -> 30.2 °C` | 全部 `0` |
| `GPU` | `60.080 s` | `590 / 9.820` | `1346.16 ms` | `47 / 65 ms` | `20 / 23 ms` | `0` | `30.2 -> 30.2 °C` | 全部 `0` |

两条路径均未触发安全中止；这只是单手机、USB 连接、短时稳定性观察。GPU 与 CPU 的
数值差异不能升级为跨平台或能效优劣结论。

## 5. QNN/HTP 诊断与生产回退

QNN 2.47.0 的 HTP FP16 capability probe 返回 `is_supported=1`，随后 delegate 尝试
启动 DSP transport，但 logcat 报告：

```text
dynamic loading failed ... libQnnHtpV75Skel.so ... errno 2
qnn_open failed, 0x80000406
Failed to load skel, error 1002
Transport layer setup failed: 14001
Failed to create device_handle for Backend ID 6, error=14001
Internal error: Failed to apply delegate
```

因此当前证据支持的结论是“QNN HTP 设备端 runtime/skeleton 与当前设备环境不完整或不
匹配”，不是“模型不存在”或“NPU 性能很慢”。正式 `PRODUCTION_ROUTE` 随后记录了
`qnn_detector_initialization_failed`，并创建 `cpu_xnnpack` detector。未重复无限重试
QNN，也未将 benchmark 中的历史 policy 字段当作本次 live 结果。

## 6. 可复算证据

本地忽略目录中的证据根为：

```text
artifacts.local/evidence/phone-admission/20260731-161634/
```

其中包括：

- `20260731-161634/benchmark.json`、`benchmark.md`、`per-image.csv`、
  `false-negatives.json`、`false-positives.json`、`risk-mismatches.json`、
  `image-assets.txt`；
- `soak-production-route.json`、`soak-gpu.json`；
- `device-getprop.txt`、`device-meminfo.txt`、`device-battery.txt`、
  `device-thermalservice.txt`；
- `device-logcat-DetectorAbBenchmark.txt`、`npu-diagnostic-logcat.txt`。

本次对 benchmark harness 做的代码变化只有：增加单模型 `BaselineOnly` 运行模式，并
让没有候选模型时的报告字段使用安全默认值；没有删除原有检查，没有允许失败，也没有
修改 App 默认模型或生产路由。

## 7. 终点与后续边界

当前终点：

```text
PLATFORM_ADMITTED_FOR_CANARY
```

适用范围：`SM-S9280 / SM8650` 手机的 LiteRT CPU 4-thread 与 LiteRT GPU delegate。

保留的子路径终点：

```text
QNN_HTP_HOLD_NOT_EVALUABLE
```

T4 仍未开始：还没有第二个固定平台，不能声称双平台公平性能比较。后续若要开展 T4，
必须先冻结同一权重和数值格式、同一输入帧/分辨率/预处理、同一解码/NMS/风险输出、同一
统计单位和预热规则。T5 调度立项预检继续关闭，不能因为本机 CPU/GPU soak 通过就实现
Android 运行时调度器。
