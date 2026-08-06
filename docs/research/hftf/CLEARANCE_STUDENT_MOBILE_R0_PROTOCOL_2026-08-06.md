# Clearance-Student Mobile R0 协议

状态：`CLEARANCE_STUDENT_MOBILE_R0_TRAINING_ONLY / DEVELOPMENT_STANDARD /
NO_DEVICE_PROFILE_UNTIL_QUALITY_PASS / NO_PRODUCTION_OR_SAFETY_AUTHORITY /
NO_TEMPORAL_HEAD_R0`

## 研究问题

在不复活 A2（完整 ViT-S 518→392）、A3（轻量时序头）或 A4（旧 RGB-D student）
的前提下，验证“真正移动端 backbone + clearance-aware metric distillation”是否能
同时降低参数/FLOPs 与端到端延迟，并保住助盲几何质量。该支线是 HFTF 的独立候选，
不改变 Canonical DA V2 518、正式 App、FRESH-TF pause 或任何生产/安全权限。

## 冻结设计

- 输入：首版固定 `384x384`（允许后续版本化为 `392x392`，不得混用）。
- Encoder：首选 EfficientViT 或 FastViT；若依赖/导出不可复现，使用明确记录的
  MobileNetV4/EdgeNeXt fallback。必须是常规 QNN/HTP 友好算子，禁止动态 attention、
  Mamba 或视频 Transformer 进入 R0。
- Decoder：紧凑 DPT-lite/SDT 风格单路径多尺度融合、depthwise 细节增强、两级上采样，
  输出 `metric_depth + confidence`，并输出 ground/camera-height/clearance 辅助量。
- Teacher：Canonical DA V2 518 仅作离线 teacher；不得读取或调参已 consumed 的
  P1/P2 truth 结果。训练使用 parent/session-disjoint development stream。
- 第一版为纯单帧任务蒸馏，不加入 temporal head、P3、tracking 或后处理覆盖。

## Loss（冻结族）

`L = log-depth + λg gradient + λs scale + λp ground-plane + λh camera-height +
λc(left/center/right clearance) + λo asymmetric occupancy + λu confidence`。

`OCCUPIED→CLEAR` 为非对称高惩罚；confidence 只能表达不确定性，不能把大量样本
覆盖成 UNKNOWN。UNKNOWN 由下游有效性、ToF 和 freshness 规则决定。

## 三臂开发屏

见 [roster](CLEARANCE_STUDENT_MOBILE_R0_ROSTER_2026-08-06.json)：C0 Canonical 518
FP16 质量基准、B0 MiDaS Small 256 FP16/W8A8 速度下界、S0 移动端 student 384 FP16。
固定使用已有 corrected prediction/truth development stream；不先建设 sealed holdout。

质量字段：raw/scale-aligned AbsRel、ground recovery、camera-height MAE、clearance
MAE、false-clear、false-block、geometry transition agreement、known coverage。

只有 S0 同时接近 C0 质量且显示明确速度优势，才可申请独立 confirmation 集。

## 量化与设备边界

顺序冻结为：`FP16 训练/开发屏 → QNN FP16 真机 profile → W8A8 QAT → 重新评价`。
禁止用 weight-only PTQ 搜索挽救质量不合格模型。设备 profile 只在开发质量屏全绿后
申请，且必须绑定模型、脚本、Git、APK/DLC、SoC/Android 与 gate receipt。任何结果
仍是 development/diagnostic，不授予默认 App、生产或安全权威。

## 停止条件与资产角色

未定义/非有限指标 fail-closed；空分母不得记为 0。质量失败保留为
`NOT_SUPPORTED` 或 `TRAINING_INVALID`，不得 retune、改阈值、换 seed 或重标后
冒充成功。产物仅写入 `artifacts.local/evidence/hftf/clearance-student-mobile-r0/`；
checkpoint/cache/receipt 可作 regression 或 negative evidence，不得作为 confirmation。
