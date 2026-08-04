# MetricTraversabilityField shadow/demo R0 协议

日期：2026-08-04（Asia/Hong_Kong）
状态：`FROZEN_IMPLEMENTATION_PROTOCOL / DEVELOPMENT_ONLY / SHADOW_DEMO_ONLY`

## 目标

把 HFTF 研究侧车的主输出从过早压缩的三带标签升级为连续方向的局部米制观测场；三带提醒只保留为独立、不可执行的末端映射。R0 必须同时保留：

- 合法尺度收据下的校正深度摘要；
- 地面平面与支持质量；
- `-40°..40°`、每 `5°` 一个方向的观测侵入距离；
- `1.0/1.5/2.0 m` 身体扫掠包络；
- class-free 侵入区域、相对候选方向和逐帧观测趋势；
- `risk_score`、`known_score`、provenance、质量和 `UNKNOWN` 原因；
- 研究 JSONL、RGB/米制深度显示资产及四联屏动态渲染。

## 非目标与权限

- 不接入默认 App、`RiskResult`、语音、震动、反馈或导航链。
- `CLEAR_OBSERVED` 只表示冻结支持规则在该帧看到了覆盖到该 horizon 的深度，不表示真实可通行或安全。
- `best_observed_clearance_direction` 固定标记为 `DEMO_CANDIDATE_NOT_SAFE_DIRECTION`。
- 无有效因果尺度锚点、锚点过期、地面失败或支持不足时 fail closed；不得从 scale-free 三带插值米制值。
- Known-height 的已消费手机结果仍为 `NOT_EVALUABLE`；Spatial Calibration Head R1 的纯 RGB 路线仍为 `0/4 folds` 失败终态；本协议不重开或救援两者。
- 输出仅为 `CANDIDATE_SIDE_LANE / DEVELOPMENT_ONLY / SHADOW_DEMO`，研究主线和默认 App 不变。

## 冻结机械规则

1. 深度必须先由当前因果尺度锚点统一缩放；过期或缺失时整个米制场为 `UNKNOWN`。
2. 地面仍用既有 `depth_ransac`，相机高度门和支持门不为展示放宽。
3. 身体半宽固定 `0.32 m`，横向余量固定 `0.10 m`。
4. 障碍高度范围固定为地面上 `0.08..2.00 m`，前向观察范围 `0.20..4.00 m`。
5. 每个方向至少 20 个支持点；侵入距离取该身体走廊障碍前向距离的 2% 分位数。
6. horizon 状态只有 `OCCUPIED_OBSERVED / CLEAR_OBSERVED / UNKNOWN_SUPPORT`。
7. 时间变化只计算同一序列、相同方向的相邻观测差，明确不解释为运动或碰撞预测。
8. AlertMapper 的演示阈值固定 `1.5 m`；`UNKNOWN` 静默，不把空值当 clear。
9. 图像质量在固定 `320×240` 灰度图上计算 Laplacian variance；低于 `20.0`，或亮度 `<=10`/`>=245` 的单侧极端曝光比例超过 `0.80`，整帧输出 `UNKNOWN_IMAGE_QUALITY`。这是保守软件门，不是标定后的感知置信度。
10. 完整研究重放启用时，原始深度与校正深度写入 ignored `artifacts.local/` 的压缩 NPZ，并在 JSONL 中保留路径、SHA-256、尺度收据、内参与质量诊断；Git 只保留合同和代码。

机器合同见 [JSON Schema](../../../schemas/metric_traversability_field_r0.schema.json)，Kotlin 隔离合同位于 `:hftf-metric-depth-canary-core`；Python producer 由 HFTF 研究包内的 `metric_traversability_field.py` 提供，不构成跨模块稳定接口。

## 验收门

- Python：连续方向、三 horizon、尺度失效、趋势、提醒隔离、显示资产和四联屏渲染测试通过。
- Kotlin：隔离数据合同、`UNKNOWN` 不可携带米制统计、AlertMapper 中央支持 fail-closed 测试通过。
- 既有 clearance/renderer 聚焦回归通过。
- `git diff --check` 与仓库 hygiene 通过。
- 结果文档不得包含新模型效果、准确率、安全或产品宣称。
