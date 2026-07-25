# THÖR source-native ID / time / transform authority recovery R1 result

状态：`INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT / VALID`

权限：`NO_INDEPENDENT_PERSON_TRAJECTORY_TRUTH_ADMISSION`

## 结论

R1 恢复了四项中的一项完整 authority：官方 Qualisys `_6D.tsv` 格式说明明确 rigid-body X/Y/Z 是刚体局部坐标原点的毫米位置，THÖR 官方 record 又把 Helmet position 定义为构成刚体的 markers 的 centre of mass。因此 R0 的 `/1000` 不再只是单位假设，`5–20m` 分母的 metric unit/reference-point 子门已闭合。

其余三项仍缺：没有 raw QTM 或逐帧 ID repair/recovery mask；冻结的 `Exp_2_run_2` 没有 paired Qualisys/LiDAR bag，也没有实测 offset/jitter；`Velodyne` mocap rigid-body 到 LiDAR measurement frame 的 lever arm/rotation/axes/handedness/extrinsic error 没有发布。唯一合法终态仍是：

`INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT / VALID`

## 四项 authority

| Authority | 判定 | 一手证据与边界 |
| --- | --- | --- |
| raw trajectory + ID/recovery provenance | `MISSING` | [THÖR 论文](https://arxiv.org/abs/1909.04403)明确说明人工清理 helmet ID switch、恢复 lost tracks，之后又自动从不完整 marker 集恢复位置；但 [people-tracks record](https://zenodo.org/records/3382145) 的 48 个文件没有 `.qtm/.qtmproj`、raw→cleaned mapping、逐帧 recovery mask 或 provenance。 |
| TSV unit + Helmet reference point | `PASS` | [Qualisys 6DOF TSV 格式](https://docs.qualisys.com/qtm/content/processing_measurement/6dof_data_format.htm)明确 X/Y/Z 是 rigid-body local-origin 的 `mm` 位置；THÖR record 明确 Helmet rigid-body position 是定义该刚体的 markers 的 centre of mass。该点不是人体 body center，也没有 anatomical lever arm authority。 |
| measured QTM—Velodyne offset/jitter | `MISSING` | 论文只有共同 NTP server 配置；[Qualisys TSV 时间说明](https://docs.qualisys.com/qtm/content/processing_measurement/motion_data_tsv.htm)反而警告 `TIME_STAMP` 可能不对应首帧，不建议用于跨设备同步。冻结 run 2 没有两侧 bag、共同物理 trigger、NTP log、offset/jitter 数值或 uncertainty。 |
| world → rigid body → LiDAR frame + error | `MISSING` | run 2 `_6D.tsv` 确有 `Velodyne` rigid-body 数值 pose、rotation matrix 与 mocap residual；但两个官方 record 均无 calibration/extrinsic 文件，缺 marker-origin→LiDAR measurement-frame lever arm/rotation、LiDAR axes/handedness和标定误差。论文的 `1mm` discretization / `2mm` mocap residual 不能外推为 sensor extrinsic error。 |

## 官方清单与 paired-bag canary

官方 people-tracks v1 发布 `13 × (MAT/TSV/6D TSV)` 加 9 个 Qualisys bag；point-clouds v1 只发布 9 个 LiDAR bag。Experiment 2 只有 `run_5` 的 paired bags：

- `ex2_run5_qualisys.bag`：`36,483,489` bytes，MD5 `e2e3592793af689ef88fccb869f97a41`，SHA-256 `2ad15eeb53a854b96a3e21f2e6fde499737bcd12bc7e7e7b2a4d0b531a32c0ae`；
- `ex2_run5.bag`：`458,220,839` bytes，MD5 `a213dc8e0ad1665bbb4e0320e203ae7d`，SHA-256 `e23580ab32cd2d49e9bb1a895031485731be503d3d6c5704d54c20887b5377ae`。

run 5 只作为格式可用性 canary，绝不替换冻结 run 2。Qualisys bag 只有 `/object_3..13` PoseStamped 与 rosout；LiDAR bag 只有 `/velodyne_packets`、robot odom、2D laserscan 和 report。两侧都没有 `/tf`、`/tf_static`、`/clock`、calibration 或 clock-diagnostic topic，且 `/object_N` connection metadata 没有 source-native body-name mapping。

run 5 的 4,633 个重叠 LiDAR header 对最近 100Hz QTM grid 的 signed residual 为约 `-4.998..+4.998ms`，标准差约 `2.888ms`。这正是异步 100Hz 采样相位可产生的范围；没有共同物理事件时不能把它解释成 inter-clock offset/jitter。

## 冻结分母与权限

R0 的 source、member、`25,912` frame、全部 `Helmet_2..Helmet_10`、`Citi_1`、整文件窗口、missing policy 和五档全部原样继承。毫米 authority 恢复后，原冻结计数保持：

`43,821 / 41,035 / 7,286 / 0 / 0`

这只说明 `0–5 / 5–10 / 10–20` 的距离数值可按米解释；由于 stable-ID recovery、run-specific clock 和完整 sensor transform/error 三门仍失败，不得把这些分母升级为 admitted independent sensor-paired truth。`20–40 / 40m+` 继续为空，`40m+` 能力边界没有删除。

本轮没有读取候选输出，没有比较 centroid、tracker 或 deskew，也没有改变序列、轨迹、窗口、缺失策略或距离带。算法比较/选择、route/event、Android、人体、独立行走和 production authority 全部关闭。

## 复算与证据

- focused tests：`3/3 OK`；
- independent rebuild validator：`24/24 VALID`；
- config SHA-256：`95884bdacad6bbc20c36acee56ec20fe9ee82a0a492475a265b688fda4439b3a`；
- acquisition SHA-256：`9b3b0a1ade212a0ddba2b4251a76e8b04fba267dff37a21b0f7f87e94e42545f`；
- source inventory SHA-256：`74bd541e6a5e7e449569e391db9b607540d6de6e4704636129196f3b113bf0bc`；
- receipt SHA-256：`e137a16424d6ae7f4ff69847dbcce4d9e05a851cc3e21b1d7df5acc3ecf44c4c`；
- validation SHA-256：`5f585c7784ceff63d327cba5c7a49d6ea46fea28a160bf22e6f266127fb118eb`。

机器证据位于 ignored `artifacts.local/evidence/thor-source-native-id-time-transform-authority-recovery-r1/`；run 5 paired-bag 只读 canary 位于 ignored `artifacts.local/work/thor-clock-authority-audit/`。

## 终止边界

R1 已完成且没有自动算法后继。只有 source owner 新发布 frozen run 2 的 raw QTM + repair/recovery provenance、paired bags/clock measurement，以及 Velodyne rigid-body→measurement-frame calibration/error，才可能另立新的 authority recovery 版本。不得换 run、截窗或用 run 5 的时间相位/frames 代替这些 authority。
