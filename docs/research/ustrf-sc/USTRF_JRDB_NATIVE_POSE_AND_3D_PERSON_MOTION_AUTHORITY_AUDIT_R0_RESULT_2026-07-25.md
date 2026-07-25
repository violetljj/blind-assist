# JRDB native pose / 3D person motion authority audit R0 结果（2026-07-25）

状态：`NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED / VALID`

权限：`P1_METADATA_COMPLETE / P1B_REQUIRED / P2_CLOSED / RISK_PRIMITIVE_CLOSED / ROUTE_EVENT_TRUTH_CLOSED / ANDROID_CLOSED / HUMAN_CLOSED / PRODUCTION_CLOSED`

## 结论

JRDB 不应因 global 2D affine 失败而被放弃。低成本审计已经证明：官方 train 版本确实存在可对齐的 RGB、双 Velodyne 点云目录、逐源 timestamp、2D/3D person track、静态 robot-camera-LiDAR transform 和同 sequence ROS bag；`cubberly-auditorium-2019-04-22_0` 的前 120 帧在这些目录/标签层全部完整，足以支持一个原生多传感器 canary 的候选窗口。

但 P1 还不能宣告完全通过。官方论文明确说明数据包含 IMU、wheel encoder，并称多传感器数据已同步；sensor setup 也说明 base-chassis 与相机/LiDAR 的 TF 位于 rosbags。当前公开下载页和 archive central directory 却没有给出 bag 内 topic inventory、message type、header clock、动态 `odom -> base_link` 覆盖或 IMU 覆盖。现有第三方 consumer 明确读取 `/tf`、`/tf_static`、`odom -> base_link -> base_chassis_link` 与双 Velodyne topic，但它只能作为可行性佐证，不能替代对原生 bag payload 的权威核验。

因此终态是：

`NATIVE_MULTISENSOR_CANARY_ELIGIBLE_POSE_IMU_TOPIC_AUDIT_REQUIRED / VALID`

唯一下一步仍属于 P1：对一个最小 train rosbag 做 topic/time/TF inventory canary。只有它确认 measured dynamic pose、IMU、双 LiDAR header timestamps 与 RGB/PC timestamp 区间可闭合，才允许另立 P2。

## 官方证据边界

- [JRDB 原始论文](https://arxiv.org/abs/1910.11792)说明传感器套件包含 6D IMU、wheel encoders、双 Velodyne、RGB，并称数据从所有传感器同步记录；2D/3D person identity 在同 sequence 内时间一致。
- [官方 sensor setup](https://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf)给出 base-chassis 到 lower/upper Velodyne 与 cylindrical camera 的 6D 变换，并说明 TF tree/individual-camera transform 位于 rosbags。
- [JRDB 官方下载页](https://jrdb.erc.monash.edu/)登录后列出 40 GB train rosbags、train/test images、pointclouds、timestamps、calibration 与 labels；未列 test rosbags，也未公布 bag topic manifest。
- [官方 benchmark 说明](https://jrdb.erc.monash.edu/benchmark/)允许从 ROS bag 提取 GPS/IMU 等信息，但明确把同步核验责任留给使用者。

这些来源足以证明“存在可审计的原生多传感器路线”，不足以证明“选定窗口的 pose/IMU 已可靠逐帧绑定”。

## 同 sequence 机器审计

冻结序列：`cubberly-auditorium-2019-04-22_0`

| 证据 | 审计结果 |
| --- | --- |
| train timestamps | 27 sequences；选定序列 `frames_img.json` / `frames_pc.json` 各 1,298 rows |
| train rosbags central directory | 53 entries、27 `.bag`；选定序列存在 |
| six RGB streams | 每路 1,298 frames；前 120 完整 |
| upper/lower Velodyne PCD | 每路 1,296 members；前 120 完整；整段缺 `001296`、`001297` |
| stitched 2D labels | 1,298 frames、38,280 objects、141 unique `label_id` |
| 3D labels | 1,298 frames、43,469 objects、140 unique `label_id` |
| RGB-PC time delta | min `-45.65 ms`、median `-1.93 ms`、max `43.50 ms` |
| static transforms | `base_link` / `occam` / `upper2ego` / `lower2upper` 合同存在 |
| dynamic pose | 官方 existence + 第三方 consumer contract；原生 payload 未审 |
| IMU | 官方 existence claim；topic/message/time coverage 未审 |

2D 与 3D track 必须通过 `sequence + frame stem + label_id` 联结；141 与 140 个 unique ID 的差异表明不能假设两侧轨迹全集完全相同。传感器必须通过 `sequence + frame stem + source timestamp` 联结；不能只按数组位置或固定 15 fps。

## 资源与复算

- producer/validator 各只读取四个 remote central directory 和两个 compressed label JSON；
- 每进程 network bytes：`35,569,929`，低于 `67,108,864`；
- 40 GB rosbag、22.3 GB images、11 GB pointcloud archive 均未完整下载；
- config SHA-256：`750e2302e4b85117743e39af213d4c2371ae6bcc55ae61f2601ee434cad221b2`
- receipt SHA-256：`c9ccd35e87ba2620cbd0b7aebad01d31ef8a46402e82d87f650225988bc6eb8c`
- validation SHA-256：`cc6adc196b8993d4ed3462cada45990aa8969f5b1c4bece223e49a9523320ffe`
- validator：deterministic recomputation、bounded network、no full archive、first-120 completeness、pose/IMU unclaimed、P2 closed、route/event/safety closed 全部通过；
- stdlib unit tests：`2/2 OK`；`py_compile` 与 scoped `git diff --check` 通过。

## P1B 唯一下一边界

建议名称：

`JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0`

只允许：

1. 选择 archive 中最小、且在 timestamps/labels/pointcloud 目录中同时存在的 train sequence；
2. 只提取 bag connection/topic index 或单个最小 bag，不下载 40 GB 总包；
3. 冻结 `/tf`、`/tf_static`、measured pose/odometry、IMU、upper/lower Velodyne 的 topic、message type、frame_id、header clock、覆盖区间、频率和 reset/gap；
4. 与 `frames_img.json` / `frames_pc.json` 的前 32–120 帧做 timestamp coverage 上界，不计算人机相对运动；
5. commanded velocity、wheel command CSV、固定帧率或第三方插值不得替代 measured pose；
6. 任一 pose/IMU/clock/frame chain 缺失则关闭 P2。

只有 P1B 通过，才可启动用户提出的最小多模态窗口 P2。即使 P2 通过，JRDB 仍只有 perception/geometry/discovery 权限，不产生 intended route、alertable/passed/cleared 或完整助盲安全验收权威。
