# BlindAssist Assistive Geometry B0 data capability and roster lock result

日期：`2026-08-09`

终态：

```text
B0_DATA_CAPABILITY_AND_ROSTER_LOCK_PASS
B1_TRAINING_NOT_AUTHORIZED
```

本结果关闭的是“是否存在一组新的、角色隔离的、实际可读取的 Assistive Geometry
研究输入”这个问题。它没有打开 student 训练，也没有把 source depth 直接宣布成可用的
ground / clearance truth。

## 1. 数据能力盘点

metadata ledger 中共有 `139` 个 RGB+depth session，`66` 个满足现有结构条件，结构通过率
`47.482%`；其中 `58/66` 的 consumed / burned / fresh / reserved 角色证据仍全为
`UNKNOWN`。因此盘点只证明“存在候选能力”，没有用路径名或缺失字段替数据分配角色。

ARKitScenes 在该盘点中 `28/28` 个现有 session 具备 readable、frame-key aligned 与 pose
结构，但最终 B0 roster 没有复用这些旧角色，也没有复用已消费 120-frame cohort 或
DepthART R2 的 8-session roster。

## 2. 冻结 roster

最终 roster 绑定 ARKitScenes 官方 metadata commit
`7283761bf26c27570ec59a5dc0f8686fbff07726`，并在仓库
`729912f4cdb0347a5e0d3d5dcd4085ddb6e7afad` 上扫描 `566` 个已跟踪研究文档，排除
`101` 个既有或源可用性失败身份。最终角色为：

| 角色 | visit | video | 用途 |
|---|---:|---:|---|
| TRAIN | 16 | 16 | 仅在 B1 协议冻结后拟合 |
| DEVELOPMENT | 8 | 8 | 仅用于候选、阈值与 evaluator 开发 |
| CONFIRMATION | 8 | 8 | 保持 sealed；候选冻结后一次性打开 |

32 个 visit 和 32 个 video 均唯一，角色间无交叉，也不与排除快照交叉。冻结后禁止换组、
补样或按 outcome 重分配。

## 3. 三个有价值的失败终态

1. Attempt 1 在 HEAD preflight 得到 `159/160`：visit `483067` / video `48018149`
   的 `lowres_wide.traj` 连续三次 HTTP 403。该身份在下一 attempt 前显式排除。
2. Attempt 2 的 HEAD 为 `160/160`，但 body integrity 在 visit `467354` / video
   `47333684` 只得到 `219` 个公共 RGB-depth-confidence 帧，低于冻结的 `300`；没有写
   manifest，也没有降低门槛。
3. Attempt 3 完成 32 视频下载，但独立 pose audit 证明 `32/32` 个“最早 300 帧”都含有
   trajectory 起点之前的帧，最差只有 `78/300` 帧可作 pose 插值。该 materialization
   永久保留为非准入证据，不用于训练或评估。

这些失败分别隔离了 URL 可用性、源包实际帧交集与帧窗口/轨迹域三个不同问题。

## 4. Attempt 4 与完整性 PASS

Attempt 4 沿用冻结 roster，不替换身份，把窗口规则改为：

> 选择同时存在于 RGB、depth、confidence，且全部位于 trajectory 时间域内的最早
> 300 个 source-native 帧。

最终实际物化 `32` 个视频、`9,600` 帧、`38,466` 个文件，共 `1,005,605,559` bytes。
独立 label-blind audit 完成：

| 检查 | 结果 |
|---|---:|
| 实际解码 RGB/depth/confidence | `28,800 / 28,800` |
| intrinsics 映射与解析 | `9,600 / 9,600` |
| 最小 depth 非零比例 | `1.0` |
| 最大 pose 插值包络 | `0.116619 s` |
| 冻结 pose 包络门 | `<= 0.25 s` |
| model output / task outcome | 未读取 |
| CONFIRMATION 用于选择 | 否 |

终态为：

```text
B0_ARKIT_POSE_COVERED_MEDIA_LABEL_BLIND_INTEGRITY_PASS
```

## 5. 权限边界

用户的数据使用授权已经覆盖该 B0 roster 的研究读取、下载、后续 truth-reader 与按角色使用，
但仍受源许可证与采集同意约束，不产生第三方数据再分发权。`UNKNOWN` 仍不是负例，
CONFIRMATION 仍不得参与校准或选择。

本轮证明了身份隔离、源资产存在、逐文件哈希、图像可解码、内参格式与 pose 时间包络；
没有证明 depth 的单位/尺度、RGB-depth 注册坐标约定、pose 插值实现、ground/clearance truth、
模型质量、task-preserving deployment 或产品安全性。因此 B1 training 继续关闭。

## 6. 唯一 successor

```text
BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_AND_REGISTRATION_LOCK
```

下一步只实现并冻结 ARKitScenes depth unit、RGB-depth/K registration、pose interpolation、
ground 与 body-swept clearance truth reader 的 exact code/hash 和验证收据。该 reader 通过后，
再单独冻结 confidence threshold、loss lambda、optimizer、epoch、batch 与 B1 arm；不得在同一
步骤中读取 DEVELOPMENT outcome 或启动 student 训练。
