# 冻结检验 v1：头戴 RGB 或 ToF 融合

2026-10-12 提交，早于检验数据的录制（计划 2026-10-13 录）。执行脚本 [frozen_fusion_eval_v1.py](host/frozen_fusion_eval_v1.py)。

## 冻结内容

- **ToF**：俯角/地面/噪声联合栈，`artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012/real_manifest.json`。投影使用段均 IMU 姿态，传感器放在头部原点；只有强档进入融合。
- **姿态门控**：俯角 > +10° 或横滚绝对值 > 25° 时不出提醒。
- **RGB**：Atom fisheye 标定（`calibration.json`），相机←ToF 旋转（`camera_tof_extrinsic.json`）。官方 DAV2-Small 相对深度，短边 518。逐帧锚定，优先级依次为：ToF（≥6 个有效 zone）→ 地面锚定 → 沿用上一帧。地面锚定的头高取 `calibration` 段的 ToF 地面拟合（限 1.0–1.9 m），没有该段时用 1.35 m。HEAD/BODY 盒子：x ±0.30 m，y −0.20…0.42 / 0.42…0.90 m，z 0.30…2.00 m；盒内像素 ≥150 即报警。
- **融合**：按 query 判定，ToF 强档 或 RGB 报警。
- 上述文件的 sha256 由脚本写入结果。

## 标签

运行前只看每段的 RGB 缩略图，写好 `labels.json`，标注规则如下：

- `calibration`：低头看地面；
- `quiet`：前方 3 m 内空通道；
- `ahead`：正前方障碍物（墙、柱子、竖杆），走近到约 0.5 m；
- `head`：头部高度横杆；
- `side`：杆从身侧约 0.6 m 处经过；
- `unknown`：其他情况。

标签写好后提交，再运行脚本。

## 判据（逐段，以就绪帧为分母）

- `quiet`：任一 query 的融合报警帧占比 ≤10%。
- `side`：融合报警帧占比 ≤30%（允许近距擦过时报警）。
- `ahead`：走近的最后 15 个就绪帧（约 3 s）中，融合报警占比 ≥50%。
- `head`：最后 15 帧中 HEAD 融合报警占比 ≥50%。

**总体主张**：所有 `ahead` 段都通过，且所有 `quiet` 段都通过。分段结果全部报告；不因结果修改参数、标签或判据。不满足判据时，如实报告“未通过”。

## 已知边界

单一用户、单一设备，标签为 Claude 判读，没有逐 query 接触真值。冒烟测试使用 2026-10-12 大厅数据（已消费），7/7 通过，只作脚本可运行的证据，见 `artifacts.local/work/frozen-fusion-eval-v1-smoke-20261012/`。
