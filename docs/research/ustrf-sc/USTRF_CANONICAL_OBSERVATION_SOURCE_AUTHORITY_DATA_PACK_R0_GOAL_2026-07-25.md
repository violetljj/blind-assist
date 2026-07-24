# USTRF canonical observation source-authority data pack R0 目标（2026-07-25）

状态：`CURRENT / AVAILABILITY_FIRST / G1_NOT_AUTHORIZED`

阶段：`CANONICAL_OBSERVATION_SOURCE_AUTHORITY_DATA_PACK_R0`

## 一、父终态与唯一问题

父 G0 已闭合为 `SOURCE_AUTHORITY_ABSENT / VALID`：当前 41/41 sequence 的 geometry/RGB/time/membership 可核验，但 canonical transform 全部 unknown，authoritative severe truncation 全部 absent。

本阶段只回答：

> 是否存在一个普通公开、许可可记录、机器人/行人视角足够相关的 source family，能在不创造 heuristic 标签的前提下，以 source-native annotation 提供逐对象 truncation/occlusion，并以 sensor/media metadata 提供 frame-bound canonical transform，从而形成一个可复算的新 data-pack canary？

这不是 G1 repair，不把新来源映射到旧 11 event，不计算 signal/slope，不重跑 C1–C3，不生成 threshold/frontier，也不开放 Android、人体或生产。

## 二、两 family 有界筛选

### A. JRDB（唯一 canary 候选）

官方资料给出的准入事实：

- 由 human-comparable social mobile robot 采集，含 stationary 与 moving robot perspective；
- 54 条序列、约 60,000 帧、360° RGB + LiDAR；
- 2D tracking label 显式包含 `truncated` 与 `occluded`，并给出逐帧 track id 与 pixel bbox；
- sensor 文档给出 base chassis、camera/LiDAR 6D transform、单相机与 stitched panorama 尺寸及 cylindrical projection；
- 官方页面标注 CC BY-NC-SA 3.0；任何本地 receipt 必须保留 non-commercial/research-only 与 no-production 边界。

metadata 证据：

- <https://jrdb.erc.monash.edu/dataset/>
- <https://jrdb.erc.monash.edu/benchmark/>
- <https://download.cs.stanford.edu/downloads/jrdb/Sensor_setup_JRDB.pdf>

先只准入 labels/calibration/metadata；在 label schema、sequence/frame identity、truncation 值域与 transform chain 完整复验前，不下载全量 RGB。

### B. nuScenes（metadata-only 对照，预期拒绝）

nuScenes schema 的 per-sample image width/height、timestamp、calibrated sensor intrinsics/extrinsics 与 ego pose 是权威 metadata；但 `visibility_token` 是对象在全部六个相机中的聚合可见比例，不是 per-camera truncation，且 vehicle perspective 与当前 human-comparable route role 不同。

因此本阶段不下载 nuScenes。除非出现官方 per-camera truncation source fact，否则以 `SOURCE_SCHEMA_AUTHORITY_INSUFFICIENT_FOR_FROZEN_ROLE` 在 metadata 门拒绝，不把 3D projection 或 visibility bin 冒充 severe-truncation annotation。

metadata 证据：

- <https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md>

## 三、JRDB canary 准入门

canary 必须在任何 route-event review 前证明：

1. 下载 URL、字节数、SHA-256、时间与许可/用途边界进入 receipt；
2. archive 路径安全，无 traversal、symlink 或 root escape；
3. label schema 的 frame、track id、truncated、occluded、bbox 与 sequence identity 可复算；
4. `truncated` 必须是 source-native annotation；若官方说明某字段可能是 arbitrary evaluation value，则必须在实际 GT 上证明不是常量/占位，否则拒绝；
5. raw camera / stitched panorama 的 frame identity 与 label identity 有官方或可验证映射；
6. camera calibration、stitch/cylindrical projection、image size 与 canonical frame 可绑定，不能假定 rotation=0；
7. timestamp/sequence continuity 与 ego/base-frame transform 在 canary 中可枚举；
8. 不将 occlusion 等同 truncation，不将 bbox 触边 heuristic 等同 source authority；
9. canary 只获得 source-authority/data-discovery 权限；route truth、event truth、acceptance、human safety 与 production 权限仍为 false。

资源门：最多 1 个 JRDB label/calibration archive canary；总下载上限 2 GiB；先 metadata/labels，未过门不下载 RGB/point cloud。

## 四、唯一合法终态

按顺序：

1. URL/access/archive/schema 无法完整复验 → `FAIL_CLOSED_ACCESS_OR_AUDIT_INCOMPLETE`；
2. truncation 是占位/任意值，或 transform/frame identity 不能绑定 → `SOURCE_SCHEMA_AUTHORITY_INSUFFICIENT`；
3. label 与 transform authority canary 通过，但 RGB/timestamp/route role 尚未物化 → `AUTHORITY_CANARY_PRESENT_ROUTE_ROLE_PENDING`；
4. label、transform、RGB/time identity 和最小 route-role availability 均通过 → `SOURCE_DATA_PACK_ADMISSIBLE_FOR_NEW_DISCOVERY`。

命中后不得选更乐观终态。即使第四态也只允许另立新 data-pack materialization 与独立 event-truth 目标；不得回填旧 11 event、直接启动 G1 或声称核心算法有效。

## 五、停止与动态优化

- JRDB 下载需登录且公开直链不可复验时，记录 exact access blocker，不尝试绕过账户或条款；
- test labels 若公开可下载，可先作为 schema canary；test RGB 不可得时不得把 labels-only 写成完整 data pack；
- truncation 对 normalized area 的 90° rotation 数值不变量与 boundary authority 分开：scale 数值合同可以在未来放宽不必要的 rotation 要求，但 truncation/boundary/G3 仍须 canonical transform；
- 两 family 后停止本轮来源扩张。不得因 nuScenes metadata 丰富而放宽 per-camera source-native truncation 门。
