# YOLO + 语义分割双环 technical smoke R0 结果

状态：`COMPLETE / VALID / TECHNICAL_ONLY / NO_EFFECT_AUTHORITY`

日期：2026-07-31（Asia/Hong_Kong）
执行者：`violjjet`

## 结论先行

本轮完成了一个独立的、非效果性的 semantic-segmentation technical smoke：

- 单一已存在 reference：`MobileNetV3Small(alpha=0.75)+LR-ASPP` INT8 TFLite；
- 24 个 RGB slot、6 个 fixed clip、3 个来源；输入仅作为排除式 technical input；
- 输入/输出 tensor 合同、有限值和 runner 可执行性通过；
- argmax 输出像素 `100%` 为 `walkable`，`boundary_step_curb`、`obstacle`、
  `unknown_nonwalkable` 均为 `0%`。

因此本轮只能得到：

> 该 reference 的接口可以运行，但在这组 RGB smoke input 上没有显示出非 walkable
> 类别输出；不能据此声称已获得有效结构分割、障碍信息或对 YOLO 的互补增量。

这不是模型跨数据集性能结论，也不是对 SegFormer、DDRNet 或其他候选的比较。

## 固定身份与输入边界

| 项目 | 值 |
| --- | --- |
| evidence instance | `DUAL_LOOP_SEGMENTATION_TECHNICAL_SMOKE_R0` |
| runner contract | [technical-smoke Module README](../../../scripts/research/dual_loop_segmentation_technical_smoke/README.md) |
| 输入 manifest | `artifacts.local/evidence/central-obstruction-agent-label-readiness-d0-a-successor-r0/calibration-input-manifest.json` |
| manifest SHA256 | `a7dad2e424373ff79b2abc506ee9ffb49c8e1e99c62d7899cf645f970149f0a4` |
| 输入角色 | `excluded_rgb_technical_input_only` |
| 样本 | 24 slot / 6 fixed clip / 3 source session |
| reference artifact | `artifacts.local/evidence/segmentation-candidate/gpu-smoke-20260713-int8.tflite` |
| artifact SHA256 | `5194871c0aec4b5c707f6b75ddaeb5ee9a526554dc229a52ae393f7f7366342f` |
| artifact size | `327,344 bytes` |

脚本只读取 fixed unit 的 `unit_id`、`session_id`、`slot_ordinal` 和
`review_image_path`，并要求 `candidate_output_visible=false`、
`prior_review_visible=false`。没有读取中央阻塞 Agent 标签、YOLO 输出、风险、反馈或融合。

## Technical smoke 结果

| 检查 | 结果 |
| --- | --- |
| 输入 | `[1,256,256,3]`, `int8`, NHWC RGB |
| 输出 | `[1,256,256,4]`, `int8`, NHWC logits |
| 输入量化 | scale `0.9333333373`, zero-point `-128` |
| 输出量化 | scale `0.0003842372`, zero-point `10` |
| dequantized finite values | `true` |
| host TFLite P50/P95/MAX | `5.2386 / 8.1098 / 12.2758 ms` |
| 设备时延 | `NOT_MEASURED` |

### Argmax 像素分布

| 类别 | 像素 | 比例 |
| --- | ---: | ---: |
| `walkable` | 1,572,864 | 100.00% |
| `boundary_step_curb` | 0 | 0.00% |
| `obstacle` | 0 | 0.00% |
| `unknown_nonwalkable` | 0 | 0.00% |

报告还明确记录了 `ARGMAX_COLLAPSED_TO_WALKABLE_ON_SMOKE_INPUT` 和
`NO_NON_WALKABLE_ARGMAX_OUTPUT_ON_SMOKE_INPUT` 两个诊断警告。

## 可视化与可复现产物

- JSON 报告：`artifacts.local/evidence/dual-loop-segmentation-technical-smoke-r0/report.json`
  （SHA256：`78ef2e641d94477df45f967b5155e651c9ec9dcaca19f5a05b9269ad9d351697`）。
- 6 个 fixed clip contact sheet：
  `artifacts.local/evidence/dual-loop-segmentation-technical-smoke-r0/visualizations/`。
- 可复现实验器、Module contract 和 5 项 focused contract tests：见
  [technical-smoke Module README](../../../scripts/research/dual_loop_segmentation_technical_smoke/README.md)。

contact sheet 的绿色覆盖与全量 `walkable` 分布一致；它是预测输出的可视化，不是
人工或 Agent 真值。

## 权限与下一步

本证据实例的 claim ceiling 是
`interface_plausibility_and_output_diagnostic_only`，并且明确：

```text
d0_a_readiness_contribution = false
d0_b_execution_authorized = false
candidate_selection_performed = false
model_comparison_performed = false
fusion_evaluated = false
device_latency_measured = false
```

因此：

- 中央阻塞路线仍保持 `CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY`；
- Q0 semantic-refresh 是旁路线，不是本轮新双环主线；
- 语义分割模型正式选型、客观互补单位、A/B/C 事件或像素效果、融合和 Android 均未完成；
- 本轮失败只说明该 reference 在当前 smoke input 上输出塌缩，不能扩大成整个分割方向失败。

后续若继续，应另立一个短、明确的 Development 设计：先冻结客观的图像空间互补单位，
再在 held-out 数据上比较 `YOLO-only` 与 `YOLO + segmentation`；不得重新使用中央阻塞
Agent 标签，也不得把本 smoke 的 `walkable=100%` 包装为风险或可通行性结论。
