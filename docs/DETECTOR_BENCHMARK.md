# 实时检测器横向评测说明

## 单目深度融合候选路线

2026-06-11 新增 `YOLO11n + 单目深度TFLite + 风险融合` 的候选评测脚手架。该路线只用于 androidTest 和本地实验，不替换 App 默认模型；正式 debug APK 仍只应包含 `assets/yolo11n_fp16_320.tflite` 与 `assets/coco_labels.txt`。

候选深度模型默认放在：
```text
.downloads/depth-lab/exports/depth_anything_v2_small_fp32.tflite
```

本机 2026-06-11 已下载官方 `Depth Anything V2 Small` PyTorch 权重到：
```text
.downloads/depth-lab/checkpoints/depth_anything_v2_vits.pth
```

候选导出入口：
```powershell
.\.venv-export312\Scripts\python.exe scripts\export_depth_anything_v2_tflite.py
```

当前状态：官方 Small checkpoint 已在 BlindAssist EvalSet 前 20 张图完成 PyTorch smoke，输出非 NaN、非全零，bbox 区域可采样，证据在 `test-artifacts.local/depth-fusion/20260611-depth-anything-v2-small-pytorch-smoke.json`。但 `onnx2tf` 转 TFLite 仍阻塞在 ONNX `wa/model/Reshape` 节点，因此尚未生成 `depth_anything_v2_small_fp32.tflite`，也尚未进入 TFLite 合同检查和真机 DepthFusion A/B。

2026-06-12 补充一个可运行的现成移动端候选：`IPDLA/MobileDepthEstimation` 仓库内的 MiDaS `depth_model.tflite`。它能通过本仓库深度模型合同检查：输入 `[1,256,256,3] float32`，输出 `[1,256,256,1] float32`，模型大小 `66,338,288` bytes；20 图 smoke 也通过，证据在：
```text
test-artifacts.local/depth-fusion/inspect-ipdla-midas-depth-model.json
test-artifacts.local/depth-fusion/smoke-ipdla-midas-depth-model.json
```

注意：该仓库没有 README / LICENSE / release，可信度和授权状态弱，只适合作为“技术链路打通候选”，不建议作为默认论文主线模型或发布资产。可用如下方式把它注入 androidTest-only benchmark 资产：
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_depth_fusion_benchmark.ps1 -DepthModelPath .downloads\depth-lab\candidate-mobile-models\IPDLA\MobileDepthEstimation-main\app\src\main\assets\depth_model.tflite -DepthModelAsset depth/depth_model.tflite -SkipDefaultRegression
```

2026-06-12 已在 `R5CX10M8Y8X` 真机完成上述 MiDaS TFLite DepthFusion A/B，证据目录为：
```text
test-artifacts.local/depth-fusion-benchmark/20260612-002427
```

结论：`do_not_promote_depth_fusion`。`candidate_depth_fusion` 的 `criticalMissCount` 从 `9` 降到 `7`，但 `alertFalsePositiveRate` 从 `0.037` 升到 `0.185`，`distanceBandAccuracy` 从 `0.73` 降到 `0.69`，`total P50/P95` 从 `54/56ms` 升到 `276/292ms`。因此该候选只证明链路可跑，不应替换默认 `yolo11n + current RiskAnalyzer`。

模型合同检查：
```powershell
.\.venv-export312\Scripts\python.exe scripts\inspect_depth_model.py
```

离线 smoke 会读取 BlindAssist EvalSet 前 20 张图，确认输出深度图不是 NaN 或全零：
```powershell
.\.venv-export312\Scripts\python.exe scripts\smoke_depth_model.py --model .downloads\depth-lab\exports\depth_anything_v2_small_fp32.tflite --dataset-root test-artifacts.local\datasets\blindassist-evalset-20260527-impl --image-limit 20
```

真机同设备评测入口：
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_depth_fusion_benchmark.ps1
```

该脚本会依次执行默认 YOLO11n 检查、深度模型 shape/dtype 检查、20 图 smoke、debug/androidTest APK 构建、正式 APK 资产隔离检查、`comparisonMode=DepthFusion` instrumentation、设备端 artifact 拉取，以及可选默认模型 90 秒真机回归。设备端 `benchmark.json` / `benchmark.md` 会比较 `baseline_geometry` 与 `candidate_depth_fusion`，重点看 `distanceBandAccuracy`、`centerRiskRecall`、`alertRecall`、`alertFalsePositiveRate`、`criticalMissCount`、`depth_ms` 和 `fusion_summary_counts`；`per-image.csv` 还会输出逐图 `fusion_summary`，用于复盘风险是几何基线、深度提升、深度拒绝还是运动趋势提升导致。

候选通过门槛：`distanceBandAccuracy` 提升或持平，`criticalMissCount` 不增加，`alertFalsePositiveRate` 不高于 baseline + `0.02`，`centerRiskRecall` 和 `alertRecall` 不下降。即使通过，也只进入下一阶段真实连续帧和长时间真机稳定性验证，不自动替换默认 App 路径。

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

如需使用 BlindAssist 专用评测集，指定 `BlindAssistEvalSet` 和评测集目录：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_detector_ab_device_benchmark.ps1 -DatasetKind BlindAssistEvalSet -DatasetRoot test-artifacts.local\datasets\blindassist-evalset-20260527-impl
```

该模式从 `manifest.jsonl` 读取真实助行风险预期字段，同时保留 AP50、precision、recall 等检测指标。设备端 `benchmark.json` 和 `benchmark.md` 会额外输出 `centerRiskRecall`、`alertRecall`、`alertFalsePositiveRate`、`distanceBandAccuracy`、`riskLevelAccuracy`、`primaryObjectHitRate`、`criticalMissCount` 和 `fusion_summary_counts`。若 `manifest.jsonl` 提供可选连续帧字段 `sequence_id`、`frame_index`、`expected_approach_state`、`expected_approach_alert`、`expected_time_to_alert_frames`，还会输出 `approachRiskRecall`、`approachFalsePositiveRate`、`approachDirectionAccuracy`、`approachCriticalMissCount`、`meanTimeToAlertFrames` 和 `approachLabeledSequenceCount`；旧评测集缺少这些字段时新增指标保持 `0`。候选模型替换建议会同时参考检测质量、融合原因和这些 BlindAssist 指标，而不是只看速度或 COCO 派生风险误差。

如需在已有 baseline 后做小步阈值扫参，可增加 `-RiskSweep`。扫参只通过 instrumentation 参数选择 benchmark 专用 `RiskAnalyzerConfig`，默认 App 行为不会因为扫参自动改变；只有在指标明确改善且误提醒没有明显增加时，才应另行把阈值提升为默认配置。

2026-05-27 已用 BlindAssist EvalSet 跑通 100 图同设备 A/B baseline，并在 baseline 后做小步阈值扫参。当前默认阈值采用 `center_near_sensitive` 的结果，将中心近距阈值从 `0.60/0.12` 小步调整为 `0.58/0.11`；该调整让默认 `yolo11n` 的 `alertRecall` 从 `0.822` 提升到 `0.836`，`alertFalsePositiveRate` 保持 `0.037`，`criticalMissCount` 保持 `9`。阈值调整后证据目录：

```text
test-artifacts.local/detector-ab-device-benchmark/20260527-175312
test-artifacts.local/detector-ab-device-benchmark/20260527-175312/device-detector-ab-benchmark/20260527-175403/
test-artifacts.local/device-regression/20260527-175552
```

核心结果如下：

| Model | AP50 | Precision | Recall | Center risk recall | Alert recall | Alert FP rate | Distance acc | Risk level acc | Primary hit | Critical miss | Total P50/P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolo11n` | 0.289 | 0.826 | 0.300 | 0.667 | 0.836 | 0.037 | 0.730 | 0.660 | 0.570 | 9 | 60/63 |
| `yolo26n` | 0.286 | 0.844 | 0.297 | 0.646 | 0.836 | 0.074 | 0.730 | 0.670 | 0.600 | 10 | 57/62 |

结论：`yolo26n` 仍不建议替换默认模型。它速度略快、precision 和 primary hit 略好，但 AP50/recall 略低，center risk recall 更低，alert false positive rate 是默认模型的约两倍，critical miss 也多 1 个。当前更稳妥的选择是保留 `yolo11n`，只采用上述 `RiskAnalyzer` 中心近距阈值小步调整。

本轮设备为 Samsung `SM-S9280` / Android 16，证据目录：
```text
test-artifacts.local/detector-ab-device-benchmark/20260527-022312
test-artifacts.local/detector-ab-device-benchmark/20260527-022312/device-detector-ab-benchmark/20260527-022419/
test-artifacts.local/device-regression/20260527-022510
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
test-artifacts.local/detector-benchmark/<timestamp>/  # benchmark 结果
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

当前本机 smoke test 结果如下，证据目录为 `test-artifacts.local/detector-benchmark/20260527-010222`：

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

本轮 Samsung `SM-S9280` / Android 16 结果如下，证据目录为 `test-artifacts.local/yolo26n-device-benchmark/20260527-015039`：

| 路径 | Runs | P50 ms | P95 ms | Min ms | Max ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pure TFLite invoke, CPU 4 threads | 100 | 36.996 | 42.060 | 36.649 | 47.831 |
| BlindAssist detector inference | 100 | 37.000 | 39.000 | 36.000 | 46.000 |
| BlindAssist detector total | 100 | 49.000 | 51.000 | 47.000 | 89.000 |

应用链路额外统计：preprocess P50/P95 为 `9/11ms`，postprocess P50/P95 为 `1/1ms`，检测数量 P50/P95 为 `2/7`，100 张 COCO 图片无失败。正式 debug APK 资产检查结果仅包含 `assets/yolo11n_fp16_320.tflite` 和 `assets/coco_labels.txt`；随后默认模型路径 `scripts/run_device_regression.ps1 -SampleSeconds 90` 通过，证据目录为 `test-artifacts.local/device-regression/20260527-015153`，冷启动 `TotalTime=736ms` / `WaitTime=738ms`。

这些结果只能说明 `yolo26n` 候选在当前设备、当前 320 FP16 导出和 COCO100 抽样下具备端侧运行与应用链路兼容性；它仍不是默认模型，也不能作为助行安全效果优于 YOLO11n 的结论。
