# 实时检测器横向评测说明

## YOLO11n vs YOLO26n 同设备 A/B 质量评测

2026-05-27 新增同设备 A/B 评测链路，用同一台 Android 设备、同一批 COCO100 图片、同一套 BlindAssist 预处理、YOLO raw 输出解析、NMS、风险分析规则和指标口径，对默认 `yolo11n` 与候选 `yolo26n` 做检测质量与风险质量对比。该流程仍不替换 App 默认模型；`yolo26n` 只进入 androidTest APK 资产，正式 debug APK 仍只包含 `assets/yolo11n_fp16_320.tflite` 与 `assets/coco_labels.txt`。

COCO100 准备脚本现在除 `coco100_manifest.json` 外，还会生成端侧评测用的标注文件：
```text
.downloads/detector-lab/datasets/coco100/coco100_annotations.json
```

推荐命令：
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_detector_ab_device_benchmark.ps1
```

默认参数为 `-ImageLimit 100 -PureWarmup 10 -PureRuns 100 -AppRunsPerImage 3 -MatchIouThreshold 0.5`。脚本会执行两模型 shape 检查、COCO100 标注检查、debug/androidTest APK 构建、正式 APK 资产隔离检查、`DetectorAbDeviceBenchmarkTest` 真机 instrumentation、设备端 artifact 拉取，并在完成后执行默认模型 90 秒真机回归。

本轮设备为 Samsung `SM-S9280` / Android 16，证据目录：
```text
test-artifacts.local-detector-ab-device-benchmark-20260527-022312
test-artifacts.local-detector-ab-device-benchmark-20260527-022312/device-detector-ab-benchmark/20260527-022419/
test-artifacts.local-device-regression-20260527-022510
```

设备端输出包含：
```text
benchmark.json
benchmark.md
per-image.csv
false-positives.json
false-negatives.json
risk-mismatches.json
```

核心结果如下：
| Model | AP50 | Precision | Recall | F1 | FP/img | FN/img | Risk FN | Risk FP | Risk flips | Total P50 ms | Total P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo11n` | 0.285 | 0.859 | 0.299 | 0.444 | 0.41 | 5.84 | 15 | 1 | 0.0 | 54.0 | 56.0 |
| `yolo26n` | 0.279 | 0.872 | 0.294 | 0.440 | 0.36 | 5.88 | 12 | 1 | 0.0 | 49.0 | 51.0 |

结论：本轮不建议替换默认模型。`yolo26n` 在当前设备上应用链路总耗时更低，误报数和风险漏报数也略好，但 AP50 与整体召回略低于 `yolo11n`，未满足“检测质量与风险稳定性不退化”的替换门槛。它可以继续保留为候选模型，后续若要进入默认模型替换评估，需要补充真实助行图片/连续帧、误报漏报人工复核、发热与长时间真机稳定性证据。

本说明记录 BlindAssist 阶段一“实时检测器横向升级”的本地评测流程。当前 App 默认检测器仍是 `app/src/main/assets/yolo11n_fp16_320.tflite`，本流程只用于候选模型导出、检查和性能 smoke test，不会替换默认模型，也不会改变运行时 `ObjectDetector` 路径。

## 本地目录

下载和评测产物默认保存在项目目录内，但不提交 Git：

```text
.downloads/detector-lab/models/      # 候选 .pt 权重
.downloads/detector-lab/exports/     # 候选 TFLite 导出
.downloads/detector-lab/datasets/    # 小型评测图片集
test-artifacts.local-detector-benchmark-*  # benchmark 结果
```

`.downloads/` 和 `test-artifacts*/` 已由 `.gitignore` 忽略，适合保存本机可复盘证据。

## 准备候选模型和 COCO8

默认候选为 `yolo26n.pt`、`yolo12n.pt`、`yolov10n.pt`，输入尺寸为 `320`，导出为 FP16 TFLite：

```powershell
.\.venv-export312\Scripts\python.exe scripts\detector_lab.py prepare --export
```

如需只评测指定候选，可重复传入 `--candidate`：

```powershell
.\.venv-export312\Scripts\python.exe scripts\detector_lab.py prepare --export --candidate yolo26n.pt
```

脚本会下载官方 COCO8 到 `.downloads/detector-lab/datasets/coco8`。COCO8 是 Ultralytics 提供的 8 图小型检测数据集，只用于确认下载、预处理、推理和结果写入链路可用；它不能代表 BlindAssist 真实助行场景。官方说明见 [COCO8 Dataset](https://docs.ultralytics.com/datasets/detect/coco8/)。

## 检查模型

默认模型仍走严格断言，期望输入 `[1,320,320,3] float32`，输出 `[1,84,2100] float32`：

```powershell
.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py
```

候选模型使用 flexible 检查，不强制默认模型契约，但会输出 shape、dtype 和输出布局：

```powershell
.\.venv-export312\Scripts\python.exe scripts\inspect_tflite.py --allow-any-shape .downloads\detector-lab\exports\yolo26n_fp16_320.tflite .downloads\detector-lab\exports\yolo12n_fp16_320.tflite .downloads\detector-lab\exports\yolov10n_fp16_320.tflite
```

## 运行 Benchmark

默认会评测当前 App 默认 TFLite 加 `.downloads/detector-lab/exports/` 下的候选 TFLite，并生成本地 JSON/Markdown 结果：

```powershell
.\.venv-export312\Scripts\python.exe scripts\benchmark_tflite_detectors.py --warmup 2 --runs 5
```

当前本机 smoke test 结果如下，证据目录为 `test-artifacts.local-detector-benchmark-20260527-010222`：

| 模型 | 后端 | 输入 | 输出 | 大小 | P50 | P95 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `yolo11n_fp16_320.tflite` | `ai-edge-litert` | `[1,320,320,3] float32` | `[1,84,2100] raw YOLO` | `5.11 MB` | `30.255 ms` | `31.422 ms` |
| `yolo12n_fp16_320.tflite` | `ai-edge-litert` | `[1,320,320,3] float32` | `[1,84,2100] raw YOLO` | `5.17 MB` | `36.089 ms` | `36.671 ms` |
| `yolo26n_fp16_320.tflite` | `ai-edge-litert` | `[1,320,320,3] float32` | `[1,84,2100] raw YOLO` | `4.73 MB` | `26.477 ms` | `26.786 ms` |
| `yolov10n_fp16_320.tflite` | `ai-edge-litert` | `[1,320,320,3] float32` | `[1,84,2100] raw YOLO` | `4.50 MB` | `36.061 ms` | `36.293 ms` |

这些数值只表示当前 Windows 本机 CPU/LiteRT smoke test，不等于 Android 真机性能，也不构成助盲安全效果结论。后续如果要替换默认检测器，必须至少补充真机端 90 秒回归、实际场景图片集和误报/漏报对照。

## 后续真实场景集

COCO8 覆盖面太小，后续建议自采或整理一组 BlindAssist 专用图片：

- 正前方近距离人、椅子、门框、桌角、箱子。
- 侧向经过的人或障碍物。
- 远处大物体与近处小物体，用于检验框大小推断相对距离的局限。
- 走廊、室内慢行、户外慢行、弱光和遮挡样本。
- 每张图片记录场景、预期主要风险方向、预期相对距离层级和是否应触发提醒。

在没有这组真实场景集前，横向评测只能判断“能否导出、能否运行、输出布局是否兼容、粗略耗时如何”，不能判断哪个模型更适合最终助行提醒。

## yolo26n 真机专项验证

本节记录 2026-05-27 对 `yolo26n_fp16_320.tflite` 的专项验证。默认 App 模型仍为 `app/src/main/assets/yolo11n_fp16_320.tflite`；`yolo26n` 只通过 androidTest assets 进入测试 APK，不进入正式 debug APK。

准备 COCO val2017 固定抽样：

```powershell
.\.venv-export312\Scripts\python.exe scripts\prepare_coco100.py --sample-count 100
```

脚本会下载官方 `annotations_trainval2017.zip`，按固定 seed `260527` 从 val2017 中抽样 100 张有实例标注的图片，写入：

```text
.downloads/detector-lab/datasets/coco100/images/
.downloads/detector-lab/datasets/coco100/coco100_manifest.json
```

专项真机验证命令：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_yolo26n_device_benchmark.ps1
```

该脚本会执行 yolo26n shape 检查、构建 debug/androidTest APK、确认正式 APK 不含 yolo26n、运行专项 instrumentation benchmark、拉取设备端 JSON/Markdown 结果，并在拉取后执行默认模型 90 秒真机回归。

本轮 Samsung `SM-S9280` / Android 16 结果如下，证据目录为 `test-artifacts.local-yolo26n-device-benchmark-20260527-015039`：

| 路径 | Runs | P50 ms | P95 ms | Min ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure TFLite invoke, CPU 4 threads | 100 | 36.996 | 42.060 | 36.649 | 47.831 |
| BlindAssist detector inference | 100 | 37.000 | 39.000 | 36.000 | 46.000 |
| BlindAssist detector total | 100 | 49.000 | 51.000 | 47.000 | 89.000 |

应用链路额外统计：preprocess P50/P95 为 `9/11ms`，postprocess P50/P95 为 `1/1ms`，检测数量 P50/P95 为 `2/7`，100 张 COCO 图片无失败。正式 debug APK 资产检查结果仅包含 `assets/yolo11n_fp16_320.tflite` 和 `assets/coco_labels.txt`；随后默认模型路径 `scripts/run_device_regression.ps1 -SampleSeconds 90` 通过，证据目录为 `test-artifacts.local-device-regression-20260527-015153`，冷启动 `TotalTime=736ms` / `WaitTime=738ms`。

这些结果只能说明 `yolo26n` 候选在当前设备、当前 320 FP16 导出和 COCO100 抽样下具备端侧运行与应用链路兼容性；它仍不是默认模型，也不能作为助行安全效果优于 YOLO11n 的结论。
