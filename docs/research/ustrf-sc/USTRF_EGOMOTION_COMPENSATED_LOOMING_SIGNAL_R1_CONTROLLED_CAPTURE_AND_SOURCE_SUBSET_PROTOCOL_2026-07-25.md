# USTRF Looming R1 受控采集与来源子集协议（2026-07-25）

状态：`PROTOCOL_FROZEN / NOT_CAPTURED / SIGNAL_CLOSED`

当前权限：`RIGID_TARGET_CAPTURE_READINESS_ONLY`

## 一、目的

本协议只把 [R1 声明级目标](USTRF_EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1_CLAIM_SCOPED_EVIDENCE_GOAL_2026-07-25.md)
落成可执行输入合同。它不采集人体、不运行信号，也不定义提醒事件。

## 二、受控装置角色

| 角色 | 合同 |
| --- | --- |
| `CAMERA_RIG` | 固定焦段、分辨率、曝光/rolling-shutter 配置的 RGB 相机，与 IMU 刚性安装 |
| `FULL_POSE_TRUTH` | 外部 mocap、独立多相机重建或同等级完整 SE(3)；C1 同时验证旋转和光心无显著平移 |
| `LINEAR_TRUTH` | 带编码器的滑轨/小车、独立激光测距或外部定位；输出相机/目标到冻结表面的米制轨迹 |
| `RIGID_TARGET` | 已知尺寸、表面模型和 marker-to-surface 变换的平面板/箱体；第一阶段不用人体 |
| `TIME_AUTHORITY` | camera、IMU、rotation/linear truth 的共同触发或测得的 offset/jitter |
| `SAFETY_CONTROLLER` | 机械限位、急停、最小间距与无人占用检测 |
| `AUDIT_STORAGE` | 原始数据留在 `artifacts.local/datasets/egomotion_compensated_looming_r1/controlled_capture/`；仓库只保存无敏感信息的 manifest/receipt |

装置型号可以后定。`CAMERA_RIG / TIME_AUTHORITY / SAFETY_CONTROLLER /
AUDIT_STORAGE` 是所有 cell 的
共同依赖；`FULL_POSE_TRUTH` 只约束 C1，`LINEAR_TRUTH / RIGID_TARGET` 只约束相应
C2 cell。某角色缺失只让依赖它的 claim/cell abstain，不能关闭无关 claim。仅有
手机 ARCore `TRACKING`、未标定陀螺仪、手持卷尺或人工口令不能获得 truth authority。

## 三、冻结动作与最小矩阵

一次 site × device × calibration × contiguous recording block 构成
`capture_cluster_id`。R1-A 至少采集 3 个独立 discovery cluster；validation 与
sealed holdout 只预注册未来必须使用全新复装/时间块，当前不采集也不分配 payload。
每个 cluster 至少一个完整 session，每个动作变体至少 2 次完整 trial。动作顺序在
录制前按固定 seed 随机化。

| Cell | 变体 | 最短有效时长 | Claim |
| --- | --- | ---: | --- |
| `PURE_ROTATION_NO_CLOSING` | yaw / pitch / roll；低/中两档角速度；相机光心平移 P95 `<=5mm` | 10s | C1 |
| `CAMERA_APPROACH_STATIC_SURFACE` | 低/中两档速度；目标静止；单调表面距离闭合 | 10s | C2 |
| `ACTIVE_RIGID_TARGET_APPROACH` | 相机静止；目标低/中两档速度主动接近 | 10s | C2 |
| `LATERAL_PASS_NO_SUSTAINED_CLOSING` | 左→右 / 右→左；两个 passing offset；包含最近点前后 | 10s | C2 反事实 |

每个 cluster 需要 28 个 trial，3 个 discovery cluster 最低 84 个 trial：

- pure rotation：每 cluster `3 axis × 2 speed × 2 repeat = 12`；
- camera approach：每 cluster `2 speed × 2 repeat = 4`；
- active target approach：每 cluster `2 speed × 2 repeat = 4`；
- lateral pass：每 cluster `2 direction × 2 offset × 2 repeat = 8`。

不得在查看信号结果后新增速度、offset 或只保留好看的 trial。

## 四、标定、同步与真值

录制前必须生成：

1. 相机内参：至少 30 帧、5 个姿态/距离桶，P95 reprojection `<=1.5px`；
2. camera-to-IMU 与 camera-to-rig 完整 SE(3)，复装平移 repeatability
   `<=10mm`、旋转 `<=1°`；
3. marker-to-target-surface 完整 SE(3) 与尺寸不确定度；
4. camera/IMU/truth 的实测 offset、jitter 和采样率；主评价要求 offset 绝对值
   `<=10ms`、jitter P95 `<=5ms`，否则单元 abstain；
5. rolling-shutter/readout；未知或补偿敏感性翻转的单元 abstain；
6. 每帧 `frame_id / source_timestamp / exposure / intrinsics_id /
   camera_pose / target_pose / direct_or_interpolated / uncertainty`。
7. `collector_agent_id / controller_run_id`、motion-program hash、限位/急停 canary
   与无人占用检测 receipt；不得建立人工采集或人工验收队列。

真值距离使用相机光心到冻结刚体表面的最近距离，不使用目标中心距离。插值只可形成
`C` 级敏感性单元；A/B 主确认仅使用 direct 或带量化不确定度的独立重建。

## 五、隔离与数据角色

- 三个 cluster 均永久为 `DISCOVERY_CONTROLLED_RIGID_R1`；
- validation/holdout 必须来自未来全新 cluster，且在各自 goal 前不得采集或打开；
- session 按复装和时间块分组；同一次录制切片不能增加独立 session/cluster 数；
- producer 必须按 arm namespace 冻结输入；R1-A base arms 只能读取 RGB 与内参，
  不得读取 truth、cell 标签、outcome 或部署 IMU；
- R1-A base arms 只读 RGB/内参；oracle namespace 可读外部 orientation/full pose，
  但不得把它与部署臂混写。部署可用 IMU 只在后继 R1-B 冻结后开放；
- audit 在 signal ledger SHA 冻结后联结 truth；
- 旧 LILocBench/CrowdBot/route-window denylist继续生效；
- 录制前补齐新 RGB 与旧数据的 decoded-pixel hash、perceptual near-duplicate
  fingerprint 和 deny receipt；不通过则禁止 producer。

## 六、来源级证据拼图

机器清单：
`artifacts.local/evidence/ustrf/egomotion_compensated_looming_r1/r1_claim_scoped_source_program_r0.json`。

- Bonn 只作为 C1 与静态表面 C2 的 `B_CANDIDATE`；先冻结真实 window、静态区域、
  OptiTrack/深度/地图 chain 和许可限制，未闭合前不进入主确认。
- REveL 只作为主动接近/横向经过 C2 的 `B_CANDIDATE`；当前 20ms 配对覆盖和 Vicon
  重投影是可用先验，但同步 offset/jitter、稳定 ID provenance 与 marker-to-surface
  不确定度未闭合的单元 abstain。
- JRDB 只作为 `C_DIAGNOSTIC`；使用 source-interpolated 轨迹、近场 `0–20m` 与
  unit-level LiDAR abstention，不能进入 C1/C2 主确认或证明人体轨迹精度。

## 七、停止条件

以下任一发生即停止采集/准入，不用补录好看样本回救：

- truth 与 signal 共享派生祖先；
- 时钟、外参、表面模型或 identity hash 漂移；
- 三个独立 session 未闭合；
- pure-rotation 平移 P95 超过 5mm；
- 同步或 rolling-shutter 敏感性使结论翻转；
- producer 读取 truth/cell/outcome；
- 需要人体、视障参与者、自由行走或产品反馈才能继续。

外部 truth marker 必须位于 signal RGB 不可见区域，或使用不会改变可见纹理/光流的
独立测量方式；否则受影响单元 abstain。

协议通过只开放受控刚性目标录制，不开放算法结果、人体、安全或生产声明。
