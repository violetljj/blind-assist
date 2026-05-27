# 本地产物目录说明

BlindAssist 的源码根目录只保留代码、文档、配置和少量明确需要版本管理的材料。设备日志、截图、临时 benchmark、真实图片评测集和历史备份统一放在本地忽略目录：

```text
test-artifacts.local/
```

该目录被 `.gitignore` 的 `test-artifacts*/` 规则忽略，默认不提交 Git。

## 目录结构

```text
test-artifacts.local/
  datasets/
    blindassist-evalset-20260527-impl/
  device-regression/
    20260525-012352/
    20260525-222502/
    ...
  detector-benchmark/
    20260527-010222/
    inspect/
  detector-ab-device-benchmark/
    20260527-021849/
    20260527-022312/
  yolo26n-device-benchmark/
    20260527-013423/
    ...
  legacy/
    test-artifacts/
    before-rebase-20260522-014530/
```

## 功能说明

- `datasets/`：真实图片评测集、数据集 manifest、YOLO/COCO 导出、QA 预览和人工复核表。当前首版 BlindAssist 评测集位于 `datasets/blindassist-evalset-20260527-impl/`。
- `device-regression/`：`scripts/run_device_regression.ps1` 生成的真机安装、冷启动、UI dump、截图、gfx/mem 和 summary 证据。
- `detector-benchmark/`：`scripts/benchmark_tflite_detectors.py` 生成的本机检测器 smoke benchmark、模型检查输出和 Markdown/JSON 报告。
- `detector-ab-device-benchmark/`：`scripts/run_detector_ab_device_benchmark.ps1` 生成的同设备 yolo11n/yolo26n A/B 评测证据、logcat、APK 资产检查和设备端 per-image 结果。
- `yolo26n-device-benchmark/`：旧 yolo26n 专项真机评测证据，保留用于历史对照。
- `legacy/`：历史迁移、rebase 前备份和早期 `test-artifacts/` 内容。只用于追溯，不作为新产物输出位置。

## 新产物默认位置

```powershell
.\.venv-export312\Scripts\python.exe scripts\build_blindassist_evalset.py
# -> test-artifacts.local\datasets\blindassist-evalset-<timestamp>\

.\.venv-export312\Scripts\python.exe scripts\benchmark_tflite_detectors.py
# -> test-artifacts.local\detector-benchmark\<timestamp>\

powershell -ExecutionPolicy Bypass -File .\scripts\run_device_regression.ps1 -SampleSeconds 90
# -> test-artifacts.local\device-regression\<timestamp>\

powershell -ExecutionPolicy Bypass -File .\scripts\run_detector_ab_device_benchmark.ps1
# -> test-artifacts.local\detector-ab-device-benchmark\<timestamp>\
```

## 维护规则

- 不把 `test-artifacts.local/` 下的原图、截图、日志、benchmark JSON 或设备拉取文件提交到 Git。
- 文档中引用证据路径时，优先使用新的分组目录。
- 需要给老师、答辩或发布包使用的 APK 仍按 `docs/APK_ARCHIVE.md` 归档，不放在 `test-artifacts.local/` 当作正式发布物。
- 清理旧本地产物前先确认是否已经有等价归档或开发日志记录，不要删除唯一验证证据。
