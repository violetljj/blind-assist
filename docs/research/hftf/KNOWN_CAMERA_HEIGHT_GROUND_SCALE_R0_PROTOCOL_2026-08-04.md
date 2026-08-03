# Known camera height ground scale R0 protocol

日期：2026-08-04

状态：`IMPLEMENTATION_AND_GATES_FROZEN / FRESH_COHORT_NOT_YET_LOCKED`

workflow：`THESIS_DEVELOPMENT`

## 决策与证据边界

本实验检验一个与 Spatial Calibration Head R1 实质不同的物理假设：不再训练网络从
冻结特征中猜测尺度，而是在当前帧的相对深度点云中拟合地面，以独立测得的相机安装
高度 `H` 恢复单一全局比例：

```text
RGB -> frozen DA V2 depth -> relative point cloud -> ground plane h_rel
    -> s = H / h_rel -> scaled depth -> left/center/right clearance
```

R0 不训练参数，不拟合 offset，不允许左/中/右独立尺度，也不读取 Metric3D、ToF、IMU、
ARCore depth 或评价真值作为运行时输入。Spatial Calibration Head R1 的 train/validation
outcome 不得用于选择本实验阈值；其 sealed 米制真值保持关闭。

本实验即使通过，也只支持“在已知相机高度、内参身份一致且当前帧存在单一可靠地面
平面时的条件性米制恢复”。它不支持全场景米制、安全产品、独立行走或默认 App 变更。

## 冻结算子

以下值直接继承既有 `evaluate_metric3d_clearance_field_a0.py` 的实现，不根据 Spatial Head
结果改动：

- 深度反投影使用 source intrinsics；主臂 stride `4`；
- ground candidates 只取图像下方 `45%`，即 `y >= 0.55 * height`；
- deterministic RANSAC seed `1729`、iterations `240`、最多 `5000` candidates；
- 平面法向必须满足 `abs(n_y) >= 0.55`；
- 至少 `100` candidates；最终至少 `80` inliers，且 inlier fraction 至少 `0.08`；
- RANSAC 与最终 admission 的距离门从旧的固定 `0.045 m` 改为唯一的新尺度无关量：
  `abs(point @ normal + offset) / h_rel <= 0.035`；`0.035` 是旧门相对典型
  `1.283588 m` SANPO camera-plane proxy 的保守等价值，不从新 outcome 选择；
- SVD 对最佳 inliers 重拟合一次；不搜索模型、seed、ROI 或 aggregation；
- `H` 必须来自独立 height receipt，范围 `[0.80, 2.20] m`，声明的绝对不确定度不得超过
  `0.05 m`；receipt 必须绑定 camera/profile/mount identity；
- `s = H / abs(h_rel)`，只接受 `[0.25, 4.0]`；整幅深度使用同一比例；
- 后续 clearance 继承现有三带、障碍高度、2nd-percentile 和 `1/1.5/2.0 m` horizons，
  不在本协议中更改。

不得用 sensor ground mask 或评价语义选择 RANSAC 点。若未来加入运行时地面分割，必须
另立候选协议；source-native ground truth mask 只能用于事后 failure stratification。

## 严格 UNKNOWN

下列任一条件触发整帧 `UNKNOWN`，不得用 raw DA、上一帧尺度、常数尺度或其他模型回填：

1. height receipt 缺失、超范围、不确定度过大或 profile identity 不匹配；
2. 图像尺寸、crop、rotation、distortion/focus/zoom 与 admitted intrinsics receipt 不匹配；
3. depth finite-positive support 不足；
4. lower ROI candidates 少于 `100`；
5. 无满足法向门的 RANSAC consensus；
6. inliers 少于 `80`、fraction 小于 `0.08` 或 normalized residual 大于 `0.035`；
7. scale 不在 `[0.25, 4.0]`；
8. 任一 clearance band 的障碍 support 不足；
9. `H +/- uncertainty` 导致该 band 在任一 horizon 上跨越 CLEAR/BLOCKED 边界；
10. source truth 后验表明楼梯、多平面或坡度使 fixed-height assumption 不成立。第 10 项
    只用于评价期把输出计为错误拒答审计，不能作为运行时 oracle。

R0 不使用跨帧平滑。时间跳变只作为冻结评价指标，避免在看到 outcome 后选择 smoother。

## 数据角色和防火墙

1. **synthetic/unit mechanics**：程序生成平面、尺度乘法、遮挡和退化输入，只验证代数、
   尺度不变性、determinism 与 UNKNOWN；不得提供效果证据。
2. **consumed debug**：历史 TUM/ARKitScenes/SANPO 只允许检查 loader、shape、坐标和 hash；
   不得汇总候选效果、查看 parent 排名或据此改门。
3. **fresh development evaluation**：在 outcome 访问前，另行 metadata-only 锁定至少
   `4` 个 parent/session-disjoint sessions；不得与 Spatial Head 24 visits 或已消费
   SANPO/TUM parents 重叠。名单固定后不得因难度、缺帧或结果换源。
4. **wearable-height confirmation**：必须另有至少 `4` 个固定相机安装、卷尺高度 receipt、
   独立内参 receipt 的新 sessions。若不存在，终态只能是
   `MECHANISM_ONLY / WEARABLE_CONFIRMATION_NOT_EVALUABLE`。

若公共 RGB-D source 的 `H` 来自其 depth/pose/floor truth，该臂必须标为
`ORACLE_HEIGHT_MECHANISM_ONLY`，不能冒充部署可获得的安装高度。

## 冻结评价

独立单元是 parent/session，不是帧。主表同时报告全体和 source-native planar-ground
opportunity strata；UNKNOWN 不从 effect denominator 消失，必须单列原因和覆盖率。

fresh development 的候选必须同时满足：

- planar-ground known coverage `>= 0.60`；全体 known coverage 仅作诊断；
- known-only clearance MAE `<= 0.25 m`；
- envelope agreement `>= 0.90`；
- false-clear rate `<= 0.05`；
- temporal delta MAE `<= 0.15 m`；
- median absolute relative scale error `<= 0.10`，p90 `<= 0.25`；
- 相对 raw DA：parent-macro MAE 严格改善，false-clear 不恶化；至少 `3/4` parents
  联合满足；
- 每个 failure stratum、最差 parent 和所有 UNKNOWN reason counts 均完整输出。

门失败后禁止更改 ROI、normalized residual、RANSAC、scale range、clearance quantile、
horizon、source、frame sampling 或 truth strata 进行救援。唯一允许的后继是先写新协议，
明确改变信息来源，例如 gravity/IMU、运行时 ground segmentation、相机条件化完整学生模型，
或尺度无关风险场。

## 终态

- 全门通过且 wearable-height confirmation 通过：
  `KNOWN_HEIGHT_GROUND_SCALE_CONDITIONAL_MECHANISM_SUPPORTED`；仍不授权生产。
- 仅 oracle-height 公共数据通过：
  `ORACLE_HEIGHT_MECHANISM_SUPPORTED / WEARABLE_CONFIRMATION_NOT_EVALUABLE`。
- coverage 不足：`GROUND_SCALE_OPPORTUNITY_INADEQUATE_NOT_EVALUABLE`。
- 有效机会充足但 effect 门失败：`KNOWN_HEIGHT_GROUND_SCALE_NOT_SUPPORTED_STOP`。
- 数据、身份、内参或高度 receipt 不合格：`HOLD_SOURCE_AUTHORITY`。

## 已知风险

- 单目 depth 的误差可能是 affine 或空间变化场，单一比例无法修复；
- 下方 ROI 可能把桌面、台阶或墙面误识别为地面；
- 无 gravity 时，相机俯仰与坡度不可辨，fixed vertical height 与平面垂距并不等价；
- 身高、姿态、镜架滑动会改变 `H`；高度误差会等比例传到所有距离；
- crop/rotation/去畸变/对焦变化会破坏内参身份；
- 既有 clearance 已在 false-clear 上失败过，尺度恢复成功不等于行动性成功。
