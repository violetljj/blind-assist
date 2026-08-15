# GA-SATOM G0

状态：`ACTIVE_PREOUTCOME_PROTOCOL / PHYSICAL_MULTIZONE_HARDWARE_NOT_PRESENT / REAL_G0_NOT_RUN / NO_SATOM_ARMS / NO_TRAINING / DEFAULT_APP_UNCHANGED`

## 稳定 Interface

- 输入必须是 physical `ST VL53L8CX 8x8` frame；每帧保留完整 64-zone 物理观测，算法只可见
  预冻结 12-zone ground-anchor pattern，即 `12/64 = 18.75%` information budget；
- 每个 zone 必须绑定 RGB 坐标系中的 sensor origin、unit ray、range、sigma/status；每帧另绑
  与 RGB 同坐标系的 unit gravity-down；
- candidate 仅对重力方向上的 range-point offsets 做固定 median consensus；机械高度真值、
  ground/non-ground label 和未来帧只进入 evaluator；
- G0 只测 ground-height observability，不加载 DepthART，不运行 SATOM policy/comparator，
  不输出 body-space clearance。

## 输出

真实入口同时接收 frozen protocol、`blindassist.ga_satom_g0.activation.v1` activation receipt
与 `blindassist.ga_satom_g0.manifest.v1`，分别 SHA 绑定 measurement JSONL 与 evaluator-only
truth JSONL，输出一个 `blindassist.ga_satom_g0.evaluation.v1` JSON。每个 parent 的 9 个 episode
各保留 300 个预注册 time slots；capture loss 必须物化为全 64-zone `INVALID` frame，不能从两条
stream 同时删除。原始传感器、标定、truth 与结果只能放在
`artifacts.local/evidence/ga-satom-g0/`。

## 安全边界

G0 PASS 最多证明：在冻结 rig、物理 sensor、surface cohort 和 information budget 下，独立
metric ground anchor 达到预注册门。它不证明 task-directed sampling、SATOM headroom、真实助盲、
Android、部署、产品、安全或论文贡献。单区 ToF4M、Bonn simulated ToF、DepthART height、人工
补齐值和旧 SATOM-R0 输出均不能替代 physical G0。

## 停止条件

当前唯一 successor 是 `GA_SATOM_G0_PHYSICAL_CAPTURE_PREFLIGHT`：采购并绑定 VL53L8CX bench、
RGB/ToF/重力外参与独立机械高度真值，在任何 range outcome access 前签署 exact fresh roster
activation。G0 任一冻结门失败即关闭 ground/body-space SATOM 家族；只有 G0 PASS 才能另立 G1，
且 G1 所有方法必须共享同一 64-zone total information budget。当前不运行 G1 或训练。
