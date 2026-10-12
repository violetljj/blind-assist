# RGB 检测区读出门槛：在带真值的 ARKitScenes 真实帧上检验

2026-10-12 · EXPLORE · 执行者 Claude。接续 [小深度模型对比](REAL_RGB_SMALL_MODEL_DEV_20261012.md) 与 [冻结检验 v1/v2](FROZEN_FUSION_EVAL_V1_PROTOCOL.md)。冻结内容不变，本文只做旁证。

## 结论

**Atom 读出事先定的门槛（盒内 ≥150 px，等于 8.3×10⁻⁴ sr）在有真值的真实室内帧上落在合理工作点：** eval 组有障碍检测区检出 92.2%（153/166），空检测区误报 4.8%（4/84）；cal 组为 98.9% / 3.0%，train 组为 90.3% / 8.6%。AUC 0.96–0.999。不修改冻结门槛。

## 做法

- **数据**：sync-rgb-tof-dataset-v1，12 个 visit、384 帧，保留原 train/cal/eval 分组。RGB 256×192，内参已知。真值为参考深度加 observed 掩码。
- **锚定**：手机 LiDAR 原生深度仿真 8×8 ToF（光轴上 45° 方形视场；取 zone 中位光轴深度，限 0.3–4 m，有效像素 ≥50%，乘性噪声 2%），用来锚定 DAV2-Small 相对深度，函数与 Atom 管线相同（`fit_affine`）。372/384 帧锚定成功。
- **检测区**：盒子定义在相机坐标系，不用重力方向（真值与预测共用同一套几何）：HEAD y −0.20…0.42，BODY 0.42…0.90，x ±0.30，z 0.30…2.00 m。
- **打分**：盒内预测立体角。参考点在盒内的立体角 ≥2×10⁻⁴ sr 记为 POS；盒内没有参考点、且盒子投影区域 ≥90% 被观测到，记为 FREE；其余记为 UNKNOWN，不计入。

各组按“FREE 误报 ≤5%”选出的门槛在 3×10⁻⁴ 到 5.7×10⁻³ sr 之间波动（FREE 只有 33–84 个），说明在这个规模上逐组选点不稳定，沿用事先门槛更稳妥。

## 局限

- 256×192 的分辨率下没有细杆，所以这里只检验一般障碍和误报；细杆能力仍以 Atom 实测为准。
- 检测区不对齐重力，ARKit 手机拿持高度和 HEAD/BODY 的身体语义不同，这里只检验“深度到盒内占据”这一步读出。
- 仿真 ToF 来自同一帧的 LiDAR，比真实 ToF 干净。
- train 组 FREE 误报 8.6% 偏高，组间差异来自场景构成，没有对它做调整。

产物：`artifacts.local/work/rgb-readout-calibration-20261012/`（`readout_rows.json` 逐帧，`readout_calibration.json`）。脚本：[rgb_readout_calibration](host/rgb_readout_calibration_dev_20261012.py)。
