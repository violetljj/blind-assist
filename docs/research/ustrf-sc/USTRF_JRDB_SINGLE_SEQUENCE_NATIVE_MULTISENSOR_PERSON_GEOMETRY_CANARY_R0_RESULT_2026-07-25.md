# JRDB single-sequence native multisensor person geometry canary R0 结果（2026-07-25）

状态：`FAIL_CLOSED_LABEL_JOIN / VALID`

权限：`IMMUTABLE_OBSERVATION_PACKET_PRESENT / MOTION_NOT_COMPUTED / ROUTE_EVENT_ALERT_ANDROID_HUMAN_PRODUCTION_CLOSED`

## 结论

`meyer-green-2019-03-16_0` 前 `000000..000119` 已形成可由第二进程完整重建的 immutable observation packet：120 张 stitched RGB、120 份 upper PCD、120 份 lower PCD、2D/3D labels、外部 timestamps、bag RGB/LiDAR header、动态 `odom -> base_link`、`imu/data` 与 `/tf_static` 全部逐帧 hash-bound。

transport、clock、双 PCD、静态 frame chain、pose/IMU interpolation 均通过冻结门；但 1,350 个 3D object-frame 中有 29 个无法唯一联结同帧 2D `label_id`。按执行前冻结的“任一 3D label 无唯一 2D join 即关闭”，终态只能是 `FAIL_CLOSED_LABEL_JOIN`。

因此本轮没有计算 person motion pair，也不能回答“人体三维运动及 robot-relative geometry 可用”。这不是 JRDB pose、IMU、点云或几何整体失败；它只否定了当前 120 帧、当前严格全量 3D→2D join 合同下的 P2 availability。

## Immutable packet 与上游门

- range-read：`124,209,382` bytes；未下载 22.3 GB images、11 GB pointcloud 或 labels 全包，未访问第二条 sequence；
- raw payload：`362` members / `110,596,529` bytes；
- packet：120/120 frame，每帧绑定 RGB SHA/3760×480 geometry、双 PCD SHA/CRC/header/POINTS、外部与 bag header timestamp、pose/IMU bracket、静态链和 label join 状态；
- upper PCD POINTS：`15,528..16,414`；lower：`12,807..14,382`；两路均与对应 bag PointCloud2 fields、width×height 一致；
- image↔PCD 最大差：`39.835904 ms <= 50 ms`；
- 外部 RGB↔bag header 最大差：`0.223 µs <= 1 ms`；外部 PCD↔bag LiDAR header 最大差：`0.232 µs <= 1 ms`；
- pose bracket / 最大单侧差：`17.742873 / 12.238146 ms <= 50 / 25 ms`；
- IMU bracket / 最大单侧差：`67.562413 / 58.479549 ms <= 100 / 75 ms`；
- upper/lower 两路独立推导的 logical RGB360→base transform translation/rotation delta 均约 `1.11e-16`；得到 `base_link <- logical RGB360` 的平移 `[-0.019685, 0, 0.9026601] m`、旋转为 identity（数值误差内）。

packet SHA-256：`6db4925540a063fa852427b8a5152a0fb40f9ff6df06a7132d0cfe359660e9ec`。

## Label join 关闭事实

前 120 帧：

| 项目 | 数量 |
| --- | ---: |
| 2D object-frame | 1,345 |
| 3D object-frame | 1,350 |
| exact joined object-frame | 1,321 |
| 3D-only | 29 |
| 2D-only | 24 |

3D-only 精确为：

- `pedestrian:17`：frame `000079..000086`，8 个；
- `pedestrian:12`：frame `000099..000119`，21 个。

2D-only 为 `pedestrian:14` 的 `000033..000056`，24 个。所有帧内部均无重复 `label_id`，所以失败不是重复键，而是跨模态覆盖不完整。

官方 source attribute 还显示 1,350/1,350 个 3D object-frame 为 `attributes.interpolated=true`，2D 为 661/1,345。本轮只把它保留为 source annotation provenance；不得把这些位置称为直接测得的人体运动。由于 label join 已先失败，audit 没有继续用 joined 子集计算 1,308 个潜在相邻 pair 或 13 条 track。

## 独立复算

- config SHA-256：`a864a2246b11157f605b0bc72d5473672acc73bc87d18ae4fd03a81661534c4e`
- materialization SHA-256：`f3817e0b376854a74da7594a2e4de0fdfa76b2d68051fb267063058a55429974`
- receipt SHA-256：`d5394ea7791fc31fda7069e7d90cb9f95ddb9aa28e3344378824a350c7367ba2`
- validation：12/12 checks true；packet 与 receipt 均由本地 raw payload + bag canonical JSON 精确重建；
- focused tests：4/4；Python compile 通过。

## 权限与下一边界

`route risk`、`event lifecycle`、提醒逻辑、Android、人体/独立行走、生产、commit 与 push 均为 false。

本冻结 canary 没有合法自动后继：不得把 1,321 个交集改成分母、忽略 29 个 3D-only、降低 join 门、换 sequence 或直接计算 motion。若未来继续，必须另立版本化的 source-label authority/join recovery 边界，并在查看新结果前说明 unmatched 3D 的合法处置与 source label interpolation 权威；正式 G1–G7 仍未开放。
