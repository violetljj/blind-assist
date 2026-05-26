# 实时检测器横向评测说明

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
