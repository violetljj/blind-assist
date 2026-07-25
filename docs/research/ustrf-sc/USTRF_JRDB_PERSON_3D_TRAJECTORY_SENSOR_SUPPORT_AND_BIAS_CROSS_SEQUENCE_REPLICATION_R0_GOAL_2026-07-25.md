# JRDB person 3D trajectory sensor support and bias cross-sequence replication R0 goal

状态：`FROZEN_BEFORE_SUPPORT_EXECUTION`

唯一问题：单一 seen-development sequence 上的 object-frame / pair LiDAR 支持率、质心残差尺度，以及远距和 3D-only 退化方向，能否在 3 个仅按 source metadata 冻结的新 JRDB train sequence 上复现？

## 冻结输入

- seen reference：`meyer-green-2019-03-16_0`，只作基线，不进入新序列选择。
- 新 sequence：`gates-basement-elevators-2019-01-17_1`、`stlc-111-2019-04-19_0`、`clark-center-2019-02-28_0`。
- 每条只取 timestamp `frames_pc` 的 logical positions `0..119`，双 PCD member stem 均为 `000000..000119`。
- 选择前未读取候选 PCD payload、label JSON payload、in-box 点数、支持率、残差或 bias ledger。
- 候选先满足 timestamp、2D/3D label member、upper/lower PCD member 与 rosbag member 的 metadata 合同；排除 Meyer Green 后按 `sha256("jrdb-cross-sequence-r0|" + sequence_id)` 排序，一次性取前 3；冻结后不替换。

完整 26-sequence 排序、ZIP central-directory、timestamp inventory、rosbag ZIP64 member 与输出路径由 [frozen config](../../../configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json) 绑定。

## 不变合同

- 原 R0 `binary_compressed` PCD 完整 LZF 解码、field-major XYZ、大小/有限点守恒不变。
- upper/lower 分别解码、变换、审计；不跨 sensor 去重，不 deskew；只做描述性 concat。
- oriented box 仍在 `logical_rgb360`，`l/w/h` 对应 local x/local y/z，闭区间查询不变。
- fused in-box `>=3` 为 `sensor-supported`；`0` 为 `annotation-only`；`1..2` 为 `abstained`；结构/非有限为 `invalid`。
- pair 仍只在双端 supported 时 supported，优先级 `invalid > abstained > annotation-only > sensor-supported`。
- gap、jump、speed、acceleration、pose sensitivity、quantile 和四类 ledger 守恒均不变。
- 分层保持 distance、occlusion、3D-only、point-support；空组为 `NOT_EVALUABLE`，禁止 `0/0 pass`。

## 报告合同

逐 sequence 报告 object-frame / pair 支持率、质心残差、motion residual 和冻结分层；worst-sequence 必须带 sequence ID 与分母；pooled 指标必须拼接 primitive rows 后复算，禁止平均 sequence 百分比或 quantile。

`81.8519%` object 支持率、`78.1437%` pair 支持率及 `0.1949/0.4807m` residual median/P95 只作为冻结参考，报告 exact delta、sequence range 和 pooled 值，不临时发明数值容差 pass gate。

远距与 3D-only 只做冻结方向性复现：

- far：`40-plus` 对比 pooled `0-20`；
- 3D-only：对比 `3d-and-2d`；
- 所有可评新 sequence 同为不利方向才是 `DIRECTION_REPLICATED`；
- 任一反向为 `MIXED_OR_CONTRADICTED`；
- 少于 2 条可评为 `NOT_EVALUABLE`。

## 权限与非目标

本轮上限为 `DIAGNOSTIC`。不改质心算法、deskew、点门或 ledger；不做候选选择、route risk、event lifecycle、alert、Android、人体/独立行走、生产、commit 或 push。
