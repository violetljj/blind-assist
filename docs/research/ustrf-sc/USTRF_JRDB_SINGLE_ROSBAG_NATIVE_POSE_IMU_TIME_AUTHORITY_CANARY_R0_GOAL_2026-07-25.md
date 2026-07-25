# JRDB single-rosbag native pose / IMU / time authority canary R0（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

## 唯一问题

JRDB train archive 中最小且同时存在 timestamps、2D/3D labels 与双 Velodyne pointcloud 的单条 rosbag，能否直接证明 measured dynamic pose、IMU、双 LiDAR 的原生 topic/message/frame/header clock 与前 120 帧外部 timestamp 覆盖闭合，从而允许另立 P2 多模态窗口？

本阶段仍是 P1B。只审原生 bag payload 与外部 timestamps，不计算人机相对运动、risk primitive、route/event truth 或 signal。

## 冻结输入

- split/sequence：train / `meyer-green-2019-03-16_0`
- 选择理由：27 条 train bag 中 compressed size 最小；`690,599,770` bytes，解压 `725,607,175` bytes
- 外部窗口：`frames_img.json` / `frames_pc.json` 的 `000000..000119`
- 获取：只 range-read ZIP central directory、local header 与该单一 compressed member；不下载 40 GB archive，不换第二条 bag
- 单次网络门：`704 MiB`

## 原生权威门

1. dynamic pose：原生 `/tf` 中 `odom -> base_link`，或原生 `nav_msgs/Odometry` 等价链；header time 非零、覆盖窗口、无倒退，且 transform/pose 不是单一常量；
2. IMU：原生 `sensor_msgs/Imu`；header time 非零、覆盖窗口、无倒退，angular velocity / linear acceleration 不是单一常量；
3. LiDAR：`/upper_velodyne/velodyne_points` 与 `/lower_velodyne/velodyne_points` 均为原生 PointCloud2，header frame/time 覆盖外部前 120 帧 pointcloud timestamp；
4. clock/frame chain：上述 header clock 与 bag record time 同一 epoch，最大绝对偏差 `<=1 s`；pose/IMU 最大 gap 分别 `<=0.5/0.25 s`，LiDAR `<=0.25 s`；
5. commanded velocity、wheel command、固定帧率、topic 名、论文 existence claim 或第三方插值均不得替代 payload。

## 合法终态

1. `FAIL_CLOSED_ACQUISITION_OR_PARSE_INCOMPLETE`
2. `NATIVE_POSE_AUTHORITY_ABSENT`
3. `NATIVE_IMU_TIME_AUTHORITY_ABSENT`
4. `NATIVE_CLOCK_FRAME_CHAIN_NOT_CLOSED`
5. `NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT`

只有第五项允许另立 P2 goal；它不直接执行 P2，也不产生 intended-route、route-event、助盲安全、Android、human 或 production 权威。
