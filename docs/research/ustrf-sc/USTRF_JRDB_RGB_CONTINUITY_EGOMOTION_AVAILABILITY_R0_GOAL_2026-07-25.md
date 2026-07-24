# USTRF JRDB RGB continuity / ego-motion availability R0（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

## 角色

这是新 JRDB source pack 的 pre-G3 短窗 availability canary，不是总目标 G3。父 G2 尚未在可修复 canonical spine 上闭合，因此本阶段不得输出 event/cell/negative 上界、ego-aware signal 或候选。

## 唯一问题

同一 JRDB test sequence 的 32 帧连续 stitched RGB，在排除 detector/GT person 区域后，是否能稳定提供 sparse LK + RANSAC 2D affine 所需的背景特征、空间分布与质量？

## 冻结窗口

- sequence：`cubberly-auditorium-2019-04-22_1`
- frames：`000000.jpg`–`000031.jpg`
- pairs：31
- timestamp source：`frames_img.json / stitched_image0`
- RGB source：官方 2019 test image ZIP64 byte-range
- labels 只用于 person exclusion mask，不读取 route/event/outcome。

## 冻结方法

- person bbox 向四周固定扩张 16 px；
- `goodFeaturesToTrack`：max 1000、quality 0.01、min distance 12、block 7；
- pyramidal LK：21×21、3 levels、30 iterations、epsilon 0.01；
- full 2D affine：`estimateAffine2D` + RANSAC，2.0 px threshold、2000 iterations、0.99 confidence、10 refine iterations；
- 4×3 grid，至少 8 cells 各有至少 3 个有效 tracked point。

每 pair 质量门：

- timestamp gap：`0 < gap <= 0.2s`；
- background detected features ≥120；
- valid tracked features ≥80；
- occupied grid cells ≥8/12；
- RANSAC inlier ratio ≥0.65；
- inlier median reprojection residual ≤1.5 px；
- inlier p95 residual ≤3.0 px；
- affine 2×2 condition number ≤10；
- determinant 在 `[0.8, 1.25]`。

短窗 availability 通过门：至少 `ceil(31×0.90)=28` pair 通过；任何失败 pair 保留原因，不用默认 ego-motion=0 回填。

## 资源门

- producer/validator 各自网络读取 ≤128 MiB；
- 单 JPEG ≤8 MiB；
- 只读一次 central directory 与 32 个目标成员；
- full archive、整 sequence、pointcloud、rosbag 均禁止。

## 合法终态

1. `FAIL_CLOSED_AUDIT_INCOMPLETE`
2. `RGB_CONTINUITY_INSUFFICIENT`
3. `EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT`
4. `SHORT_WINDOW_EGOMOTION_AVAILABILITY_PRESENT`

成功只允许扩大 source-availability 审计；不授权 G3、G4、route truth、signal、Android、human 或 production。

