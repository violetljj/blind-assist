# JRDB single-sequence native multisensor person geometry canary R0（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

最大权限：`OFFLINE_SOURCE_NATIVE_GEOMETRY_AVAILABILITY_ONLY`

## 唯一问题

在父 P1B 已证明原生 pose、IMU、LiDAR header clock 可闭合后，JRDB train 的 `meyer-green-2019-03-16_0` 前 `000000..000119` 能否形成一个由 stitched RGB、双 PCD、2D/3D `label_id`、动态 `odom -> base_link`、`imu/data` 和静态 transform 共同绑定的 immutable observation packet，并支持 source-native person 3D motion 与 robot-relative geometry availability？

本阶段不定义 route、risk、event、alert、人体安全或产品行为。

## 冻结输入与帧语义

- 唯一 sequence/window：`meyer-green-2019-03-16_0 / 000000..000119`；不换 sequence、不扩窗口；
- RGB：`images/image_stitched/<sequence>/<frame>.jpg`；
- PCD：`pointclouds/{upper,lower}_velodyne/<sequence>/<frame>.pcd`；
- label：`labels_2d_stitched` 与 `labels_3d`，join key 为 `sequence + frame stem + label_id`；
- 3D label 使用官方 toolkit 的 logical RGB360 metric frame：原生点云先按 `calibration/defaults.yaml` 的 upper/lower-to-RGB 变换进入该 frame；
- 动态 pose：bag `/tf` 的 `odom -> base_link`；IMU：`imu/data / ext_imu_frame`；静态链必须直接来自 bag `/tf_static`；
- source-native motion 只把同 `label_id` 的相邻帧 3D center 经 pose 转入 `odom` 后求差；robot-relative geometry 只报告 `base_link` 中位置与距离。

## Immutable observation packet

物化器只 range-read 三个官方 ZIP 的 central directory 与上述 362 个成员（120 RGB、240 PCD、2 label JSON），逐成员校验 ZIP CRC、大小与 SHA。packet 逐帧绑定：

1. RGB/PCD 路径、SHA、尺寸或 PCD header/point count；
2. image、upper/lower PCD source timestamp 与 bag LiDAR header；
3. 2D/3D label_id 及 3D center；
4. pose/IMU 两侧原生 sample、插值权重与插值值；
5. bag `tf_static` frame chain、calibration SHA 与推导后的 logical RGB360-to-base transform；
6. 所有原始 payload、父 receipt/config 和软件产物的 hash。

packet 写入后 validator 必须从本地 raw payload 与 bag 独立重建并做 canonical JSON 精确比对；audit 只能消费已冻结 packet。

## Fail-closed 门

- 120 帧任一 RGB、upper/lower PCD、timestamp 或 label frame 缺失即关闭；
- image 与任一 PCD timestamp 差 `>50 ms`，或外部 RGB/PCD 与对应 bag header 差 `>1 ms` 即关闭；所有匹配必须一对一且单调；
- pose bracket `>50 ms` 或单侧 `>25 ms`；IMU bracket `>100 ms` 或单侧 `>75 ms` 即关闭；
- `/tf_static` 必需边缺失，或 bag 静态链与官方 upper/lower-to-logical-RGB calibration 在 `1e-5 m / 1e-5` 内不能闭合即关闭；
- 同帧 `label_id` 重复，或任一 3D label 无唯一 2D join 即关闭；
- 非有限坐标、空 PCD、非法 JPEG/PCD header 或任何 hash/CRC 漂移即关闭。

availability 通过还要求：至少 32 个 joined-person frame、31 个合法相邻 motion pair、1 条 motion track。该门只判断 source-native geometry 是否可算，不判断 motion 是否危险、是否位于路线或是否应提醒。

JRDB label 自带的 `attributes.interpolated` 是 source annotation provenance，本轮逐项报告，但不把它与本 canary 新执行的 pose/IMU 时间插值混为一谈。它没有独立的插值时间上界字段，因此不得被改称直接传感器测量；本阶段的 interpolation hard gate 只适用于 packet 为每一帧执行的 pose/IMU bracket。若 label join 先失败，则不得继续计算任何 motion。

## 合法终态

1. `FAIL_CLOSED_ACQUISITION_OR_PACKET_INCOMPLETE`
2. `FAIL_CLOSED_CLOCK_BINDING`
3. `FAIL_CLOSED_STATIC_FRAME_CHAIN`
4. `FAIL_CLOSED_POINTCLOUD_FRAME_MISSING`
5. `FAIL_CLOSED_LABEL_JOIN`
6. `FAIL_CLOSED_INTERPOLATION_BOUND`
7. `PERSON_GEOMETRY_AVAILABILITY_INSUFFICIENT`
8. `SOURCE_NATIVE_PERSON_GEOMETRY_AVAILABLE`

任何失败均保留精确 gate 与不可变证据；不得降低上界、缩分母、改 join 语义或换 sequence 回救。

## 明确非目标

不得读取或生成 intended-route、route risk、event onset/alertable/clearance、提醒策略、Android runtime、人体/盲人独立行走、生产或发布权威。成功也只允许说明本 sequence/window 的 source-native geometry availability。
