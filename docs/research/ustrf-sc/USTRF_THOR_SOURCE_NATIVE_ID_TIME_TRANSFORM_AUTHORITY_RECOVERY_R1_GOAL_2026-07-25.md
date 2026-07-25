# THÖR source-native ID / time / transform authority recovery R1

状态：`FROZEN_BEFORE_SOURCE_AUDIT`

## 唯一研究问题

在不读取任何候选输出的前提下，THÖR 官方 raw QTM、paired bag、标定文件和格式说明是否能同时闭合：

1. 原始轨迹与逐帧人工 ID 修复 / lost-track recovery mask；
2. TSV 明确单位与 Helmet rigid-body translation reference-point；
3. 冻结 run 的 QTM—Velodyne 实测时钟 offset / jitter；
4. mocap world → Velodyne rigid body → LiDAR measurement frame 完整外参、轴/手性/lever arm 与量化误差。

## 冻结与非目标

继承 R0 的 `Exp_2_run_2_6D.tsv`、全部 `Helmet_2..Helmet_10`、`Citi_1`、整文件窗口、全零 missing sentinel、禁止插值/改 ID/换 run/截窗，以及 `0–5 / 5–10 / 10–20 / 20–40 / 40m+` 五档。不得读取候选输出，不比较 centroid、tracker 或 deskew。

本轮只审计官方 source-native artifacts。论文中的共同 NTP 配置不等同于实测 offset/jitter；QTM 中 `Velodyne` rigid-body pose 不等同于 rigid-body marker frame 到 LiDAR measurement frame 的外参。

## 唯一终态

- 四项全部闭合：`INDEPENDENT_PERSON_TRAJECTORY_TRUTH_SOURCE_ADMITTED`
- 任一关键 authority 缺失：`INDEPENDENT_PERSON_TRAJECTORY_TRUTH_AUTHORITY_ABSENT`
