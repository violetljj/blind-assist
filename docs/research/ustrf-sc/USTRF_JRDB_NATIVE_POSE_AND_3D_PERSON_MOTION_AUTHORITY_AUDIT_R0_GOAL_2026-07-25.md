# JRDB native pose / 3D person motion authority audit R0（2026-07-25）

状态：`FROZEN_BEFORE_EXECUTION`

## 唯一问题

JRDB 完整公开版本是否存在足够强、可按同一 sequence/frame/time 绑定的原生证据，使一个后继短窗能够同时消费 robot pose/odometry、IMU、双 Velodyne 点云、RGB、2D/3D person track 与 robot-camera-LiDAR transform？

本阶段只做 metadata、archive central directory、timestamp/label schema 和 calibration/consumer-contract 审计。禁止下载完整 image、pointcloud 或 rosbag archive；禁止把 global 2D affine 作为主要 ego-motion；禁止进入多模态运动计算、风险原语、route/event truth、Android、human 或 production。

## 冻结 canary

- split/sequence：JRDB 2019 train / `cubberly-auditorium-2019-04-22_0`
- 后继窗口上界：前 120 帧；本阶段只审目录与 join key
- 主要 join：`sequence + 6-digit frame stem + source timestamp`
- person join：逐 sequence 稳定 `label_id`，不得按数组位置联结
- pose join：必须来自 rosbag 内带 timestamp 的动态 TF/odometry；commanded velocity 不得替代 measured pose
- IMU join：必须核验实际 rosbag topic、message type、header clock 与覆盖；论文中的“包含 IMU”只能证明 existence claim

## 通过语义

P1 只有在 pose/IMU 的实际 topic inventory 与时间覆盖也被核验时，才允许判为可直接进入 P2。若 RGB/LiDAR/labels/calibration/rosbag directory 已闭合，但 pose/IMU 仍只有论文声明或第三方 consumer contract，合法终态必须停在 `NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED`，下一步仍属于 P1，而不是 P2。

## 合法终态

1. `FAIL_CLOSED_AUDIT_INCOMPLETE`
2. `NATIVE_MULTISENSOR_BINDING_ABSENT`
3. `NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED`
4. `NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_PRESENT`

成功也不产生 intended-route、route-event 或安全真值权威。
