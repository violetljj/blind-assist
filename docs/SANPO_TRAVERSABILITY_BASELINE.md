# SANPO 可通行区域 / 通用障碍候选基线

## 结论

首版采用两阶段候选通道，不改生产 APK 的默认 `yolo11n_fp16_320.tflite`：

1. **SANPO mask oracle**：先用 SANPO 真值分割验证通行走廊、路沿、台阶、通用障碍、杆状物等规则与指标。
2. **移动分割模型**：规则通过后，自行训练并导出 INT8 TFLite，以完全相同的输出接口替换 oracle。

这样可以分别回答“风险规则是否有效”和“模型分割是否足够准”，避免模型误差掩盖规则问题。

## 为什么不能直接下载一个 SANPO 分割模型

SANPO 官方论文给出的端侧分割基线是量化 MobileNetV3、512×512、三类输出（safe / not safe / obstacles），论文报告 Pixel 6 可达 30 FPS。但截至本基线建立时，SANPO 官方数据仓库没有列出这套三分类权重。Project Guideline 公开的 `guideline.tflite` 是紫色引导线分割，不是 SANPO 通用语义分割；其 `depth.tflite` 可作为后续几何证据，但不能代替区域语义。

官方资料：

- [SANPO WACV 2025 论文](https://openaccess.thecvf.com/content/WACV2025/papers/Waghmare_SANPO_A_Scene_Understanding_Accessibility_and_Human_Navigation_Dataset_WACV_2025_paper.pdf)
- [SANPO 补充材料（含完整标签与三类映射）](https://openaccess.thecvf.com/content/WACV2025/supplemental/Waghmare_SANPO_A_Scene_WACV_2025_supplemental.pdf)
- [SANPO 官方数据仓库](https://github.com/google-research-datasets/sanpo_dataset)
- [Project Guideline 模型目录](https://github.com/google-research/project-guideline/tree/main/project_guideline/vision/models)

## BlindAssist 映射

官方三类映射把 `stairs` 和 `terrain` 归入 safe-to-walk。BlindAssist 不直接照搬：

- `stairs` 保留为明确的台阶风险；
- `curb`、`inaccessible surface`、`obstacle`、`pole` 等保留为通用风险区域；
- `terrain` 仅作为候选可通行面，后续必须由深度突变、坡度或时序证据否决；
- 大面积 building/tree/vegetation 不直接触发提醒，只在通行走廊内形成近场占据时参与风险。

代码中的 `BlindAssistSanpoTaxonomy`、`TraversabilitySegmentationAnalyzer` 会输出：

- 中心梯形走廊的 safe / not-safe / obstacle 覆盖率；
- 位于走廊内、达到最小面积的通用风险框；
- `DetectionSource.SEGMENTATION` 来源标记，避免把分割区域伪装成 COCO 检测结果。

## 运行 oracle 基线

离线 mask 分析：

```powershell
.\.venv-export312\Scripts\python.exe scripts\benchmark_sanpo_traversability.py `
  --dataset test-artifacts.local\datasets\blindassist-sanpo-pilot-20260711
```

真机风险 A/B（同一 YOLO，区别仅在是否加入 SANPO oracle 区域）：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_detector_ab_device_benchmark.ps1 `
  -DatasetKind BlindAssistEvalSet `
  -DatasetRoot test-artifacts.local\datasets\blindassist-sanpo-pilot-20260711 `
  -ComparisonMode SanpoTraversabilityOracle `
  -ImageLimit 30
```

oracle 区域只进入风险分析，不进入 YOLO 的 AP/precision/recall 统计。

当前 30 帧本地 pilot 的离线 oracle 结果（mask 最近邻缩放到 512×512）：中心走廊 30/30 帧存在候选风险，人工指定的主分割区域有 26/30（86.67%）被同口径四连通域规则覆盖；走廊平均像素构成为 safe 71.22%、not-safe 15.26%、obstacle 13.53%。该结果只证明规则能覆盖当前单序列，不代表真实模型召回率，也不能作为上线依据。

## 2026-07-11 真机 A/B

设备：Samsung SM-S9280（同一设备、同一 30 帧、每帧 3 次）。默认模型 90 秒回归通过。

| 指标 | YOLO 几何基线 | SANPO oracle 候选 |
| --- | ---: | ---: |
| 错误提醒率 | 3.3% | 90.0% |
| 距离档准确率 | 3.3% | 10.0% |
| 风险等级准确率 | 3.3% | 10.0% |
| SANPO 主风险命中 | 0% | 10.0% |
| 逼近召回 | 0% | 40.0% |
| 逼近误报率 | 0% | 33.3% |
| 总延迟 P50 | 53.0 ms | 92.404 ms |
| 总延迟 P95 | 53.0 ms | 96.678 ms |

结论：`do_not_replace_default_model`。当前序列 30 帧均标注为不应提醒；候选在第 1–26 帧持续选择右侧 `curb` 并升到 `MEDIUM/NEAR`，造成主要误报。最后 3 帧选择中心 `generic obstacle`，但保持 `LOW` 且未提醒。下一轮应把路沿改为“边界证据”而不是独立障碍框：没有中心走廊侵入、近场深度突变或连续逼近证据时，不得从 LOW 升到提醒级；分割证据的单帧晋级也应限制为最多一级。完整真机报告位于 `test-artifacts.local/detector-ab-device-benchmark/20260711-153543/`。

## Traversability v2 第一阶段结果

2026-07-11 在 Samsung SM-S9280、同一 30 帧、每帧 3 次条件下复测。路沿不再作为普通障碍 Detection；通用障碍使用完整连通域中心重叠、底部位置与中心优先排序；mask 从 512×512 降到 256×256并复用工作缓冲和同帧解码结果。

| 指标 | v1 Oracle | v2 Oracle |
| --- | ---: | ---: |
| 错误提醒率 | 90.0% | 3.3% |
| SANPO 主风险命中 | 10.0% | 86.7% |
| 总延迟 P95 | 96.678 ms | 65.919 ms |

YOLO 指标无退化，benchmark 判定 `traversability_rules_ok_for_model_stage`。该结论只代表当前否定集通过；公开正负连续序列完成前仍不训练、不替换模型。证据目录：`test-artifacts.local/detector-ab-device-benchmark/20260711-163629/`。

## 下一阶段模型契约

建议训练 `MobileNetV3 + LR-ASPP` 或轻量 DeepLab 解码头，512×512、INT8，输出至少四个风险头：

1. `walkable`
2. `boundary_step_curb`
3. `obstacle`
4. `unknown_nonwalkable`

训练顺序为 SANPO-Synthetic 预训练、SANPO-Real 微调；另建中国场景回归集，覆盖路桩、低矮台阶、盲道占用、电动车和临时施工。晋级门槛看近场危险召回、台阶/路沿召回、错误提醒率、时序抖动和端侧 P95 延迟，不以总体 mIoU 单独决定上线。
