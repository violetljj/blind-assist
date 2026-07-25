# JRDB single-rosbag native pose / IMU / time authority canary R0 结果（2026-07-25）

状态：`NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT / VALID`

权限：`P1B_COMPLETE / P2_SEPARATE_GOAL_MAY_BE_FROZEN / P2_NOT_EXECUTED / ROUTE_EVENT_SAFETY_CLOSED / ANDROID_HUMAN_PRODUCTION_CLOSED`

## 结论

JRDB train 的最小合格 bag `meyer-green-2019-03-16_0.bag` 已直接证明原生 measured pose、IMU、双 Velodyne 与外部 RGB/pointcloud timestamp 使用可闭合的 epoch/header clock。P1B 因此通过，允许另立一个同 sequence、前 32–120 帧的 P2 多模态 perception/geometry canary。

这不是 P2 结果，更不是 intended-route、alertable/passed/cleared、助盲安全或生产权威。当前只证明 JRDB 可提供稳定 ego pose/IMU 和时间绑定；不得把 robot 前向轴、wheel command 或 3D person motion 自动解释为用户路线与安全事件。

## 单 bag 原生证据

bag 共 `85,527` messages，录制 `31.999 s`。冻结外部窗口为 `000000..000119`，覆盖 `1552771939.3299458..1552771947.46275 s`。

| 角色 | 原生 topic / frame | 全 bag samples | 窗内 samples | 最大 gap | bag-header 最大差 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dynamic pose | `tf: odom -> base_link` | 3,183 | 849 | 17.807 ms | 41.996 ms | 3,183 个不同 transform；0 倒退 |
| IMU | `imu/data`, `ext_imu_frame` | 622 | 166 | 67.562 ms | 359.871 ms | 622 个不同测量；0 倒退 |
| upper LiDAR | `upper_velodyne/velodyne_points` | 471 | 125 | 70.774 ms | 10.173 ms | 外部 timestamp 最近差最大 0.232 µs |
| lower LiDAR | `lower_velodyne/velodyne_points` | 478 | 127 | 71.129 ms | 309.882 ms | 外部 timestamp 最近差最大 0.224 µs |

两个原生 odometry topic 也独立闭合 `odom -> base_link`：`segway/feedback/wheel_odometry` 3,184 条、`segway/odometry/local_filtered` 3,179 条，均 0 倒退、最大 gap 17.807 ms。权威判定优先使用直接审到的动态 TF；commanded velocity topic 未参与。

ROS1 bag 中 topic 名没有前导 `/`。审计器按 ROS 名等价规则只去掉前导 `/`；不同 odometry/IMU topic 分开审计，禁止合并后制造伪倒退或伪频率。

## 获取与复算

- 27 个 train bag 中 frozen member 的 compressed size 最小：`690,599,770` bytes；解压 `725,607,175` bytes；
- 只 range-read 6,286-byte central directory、local header 和该成员，network `690,606,150 / 738,197,504` bytes；
- 未下载 40 GB full archive，未读取第二条 bag；
- ZIP CRC32 `56,443,065` 与 central directory 一致；
- bag SHA-256：`cfa972baa5936935d4e54be54d03788155c22fec206988619226f8e566d30a09`
- config SHA-256：`8e07cef2b857c05633e121a8e05063485cc71481bc13d732a584b2f69fa12b58`
- acquisition SHA-256：`ae0874b90abf16fa49160e6f4edea1623636e462301ee5773b214ec5233f54d6`
- receipt SHA-256：`c3501d733a329decd96b43a4431da5848803d6c4eed0f5c9d05ac64b8b041004`
- validation SHA-256：`a382d6b734b7efb591a28987e3db5d3d4356bd7ebcaaa3ead59e8c310aa334b5`
- 第二进程完整重解码并逐字段比对，9 项 validation checks 全通过；
- 3 项 stdlib tests、Python compile、docs index、tracked 与新文件 scoped diff checks 通过；
- 全库 `check_repo_hygiene.ps1` 仍被 8 个既有旧 config 直接引用 research Implementation path 阻塞；报错文件均不在本任务范围，本轮未扩张修复。

## 下一边界

唯一可继续边界是另立 JRDB 单 sequence P2：

1. 冻结 `000000..000119` 的 RGB、双 PCD、2D/3D `label_id`、动态 pose、IMU 与静态 transform join；
2. 先物化 immutable observation packet，报告缺帧、插值、frame transform 与 pose/IMU quality；
3. 只计算 source-native person 3D motion 与 robot-relative geometry availability，不定义 route risk 或提醒事件；
4. 任一时钟、frame chain、label-ID、点云缺帧或插值上界失败即关闭 P2。

正式 G1–G7、route truth、event lifecycle、Android、human、production、commit 与 push 仍未授权。
