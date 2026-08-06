# Clearance-Student Mobile S1 协议

状态：`S1_A_GEOMETRY_ONLY_TRAINING / DEVELOPMENT_FAST_SCREEN /
S1_B_BLOCKED_UNTIL_S1_A_GEOMETRY_PASS / NO_QNN_PROFILE_YET /
NO_PRODUCTION_OR_SAFETY_AUTHORITY`

S1 是 S0 的机制性 successor，不是 S0 的调参修复。S0 已归档为
`NOT_SUPPORTED`：其 1.27M 模型联合优化后虽然 false-clear 极低，但
scale-aligned AbsRel、camera-height、clearance、collision agreement 和 geometry
transition 均失败。S1 通过更大容量、四尺度 decoder、feature distillation 和课程训练
测试不同机制。

## 固定设计

- 输入：`384x384`，后续如改为 `392x392` 必须新版本。
- Encoder：首版使用 torchvision MobileNetV3-Large（约 5–10M 目标容量，常规卷积/
  depthwise/QNN 友好），不并行开启 MobileNetV4/EfficientViT/FastViT。
- Decoder：四个 encoder taps、lateral projection、depthwise-separable refinement、
  progressive upsampling 和正式 depth head；保留 geometry pooled heads。
- Student taps：1/4、1/8、1/16、1/32 四尺度；每级投影到 64 channels。
- Teacher：Canonical DA V2 518 offline。R0 已有 teacher depth cache 可复用；feature
  distillation 使用 teacher DPT encoder 的冻结中间层，若缓存未物化则 runner 必须显式
  失败，不能以 depth-only 冒充 feature distillation。

## 两阶段课程

### S1-A：几何骨架

只优化 teacher log-depth、gradient、scale、feature distillation、ground-plane 和
camera-height。S1-A 的快速屏只看：scale-aligned AbsRel、camera-height MAE、clearance
MAE、collision agreement、false-clear、false-block；六项均须有限并记录分母。

若 clearance/camera-height 仍严重落后 Canonical，终止 S1-A，不进入 S1-B。

### S1-B：助盲任务头

只有 S1-A 基础几何屏通过才允许从 S1-A checkpoint 继续训练；加入 clearance、occupancy、
confidence，false-clear 非对称权重保持温和。若 false-clear 极低但 false-block 或
collision agreement 塌缩，终止当前 loss 族。

## Feature distillation

`L_feature = Σ_l || P_l(F_student_l) - stopgrad(F_teacher_l) ||_1`，固定四个尺度、
固定投影层和权重，不做 feature/weight/seed 搜索。teacher feature cache 必须绑定
teacher checkpoint、source manifest、protocol、shape 和 SHA256。

## 权限与终止

开发屏通过前不做 QNN FP16 profile；通过后也只允许申请独立的 `S1-Q` FP16 profile，
不自动进入 QAT。所有结果是 development/diagnostic，不改变默认 App、Canonical、
FRESH-TF pause 或生产/安全权限。未定义指标 fail-closed；不得在 consumed 120 帧上
retune 或重选 checkpoint。
