# JRDB person 3D trajectory far-range denominator adequacy metadata-blind replication R0 goal

状态：`FROZEN_BEFORE_CANDIDATE_LABEL_OR_PCD_PAYLOAD`

权限上限：`DIAGNOSTIC`

## 唯一问题

在不根据 label 距离、PCD 支持或 residual 选择 sequence/window 的前提下，`40m+` object-frame 支持率下降能否在至少 3 条分母充分的未见 JRDB sequence 上同向复现？

## 冻结顺序

1. A 进程只读 `frames_pc` timestamp、labels/pointclouds ZIP member inventory 和 rosbag member metadata。
2. 排除 Meyer Green、上一轮 Gates/STLC/Clark 与已读 payload 的 Cubberly；按固定 sequence hash 排序、固定 window hash 取 8 条未见 sequence × 360 个连续 frame，一次冻结且不得替换、挪动或扩窗。
3. B 进程只读冻结窗口的 2D/3D label；每条必须同时有 `>=100` 个 `40m+` 与 `>=100` 个 pooled `0-20m` valid-3D object-frame，至少 3 条通过。否则终止 `DENOMINATOR_INSUFFICIENT`，不得读取 PCD。
4. 仅门通过后，C 进程对所有充分 sequence 运行 hash-frozen support kernel；D 进程重建 freeze、gate、aggregate、四类守恒和终态。

`n=100` 对单 sequence 二项比例给出最坏情形约 `±9.8pp` 的 normal-approximation 95% half-width；主要独立单位仍是 sequence，而不是 pooled object-frame。

## 算法与报告不变量

- PCD 继续采用完整 LZF、field-major little-endian float32 XYZ 解码与 finite-point 守恒；
- upper/lower 分别变换和审计，不去重、不 deskew；
- logical-rgb360 oriented-box、闭 half-extents、`>=3` fused 点支持门不变；
- centroid 仍是 box 内所有 fused XYZ 的逐轴算术均值；
- object/pair 仍只有 `sensor-supported / annotation-only / abstained / invalid` 四类，pair precedence、quantile、motion、pose sensitivity 和 range/point 分层不变；
- pooled 指标由 primitive row 拼接后复算，不平均 sequence 百分比或 quantile；
- 主门只判断 `40m+` 相对 pooled `0-20m` 的 support direction；同步报告 3D-only、遮挡、零点、1–2 点、3–9 点与 10+ 点分母和 residual。

## 明确非目标

不改 centroid，不做 robust centroid 或 deskew，不比较/选择算法，不做 route、event、alert、Android、人体、独立行走或生产推断。即使方向复现，也只说明 frozen annotation-conditioned LiDAR support 的距离退化，不是 independent person-center/trajectory truth。
