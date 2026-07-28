# RCLE natural-session expansion Discovery R0 结果

日期：2026-07-28
终态：`STANDALONE_ROTATION_ROUTE_STOP / VALID_MULTI_SESSION_DESCRIPTION`

## 结论

metadata-only 冻结的四个 Discovery/Development capture session 已用未改 R3
完整运行；ADVIO sequence16 保持 `SEALED_UNSEEN`，没有下载、解压、内容访问或算法
调用。

预冻结停止规则要求：至少两个 session 在各自 source-pose 角速度最高 20% 的固定
pair 层中，同时出现 compensated 三连续触发密度高于 raw、且 compensated
absolute response 中位数高于 raw。结果为：

- sequence13：恶化；
- sequence14：未恶化；
- sequence15：恶化；
- sequence17：恶化。

因此 `3/4` 个 session 满足条件，正式结束 standalone rotation 路线。下一机制诊断
转向步态振荡、运动模糊、低纹理和 flow-quality gate；本结果不授权立即实现
reference-track、temporal consistency、bearing 或 Android。

## 冻结与执行

[冻结合同](RCLE_NATURAL_SESSION_EXPANSION_DISCOVERY_R0_CONTRACT_2026-07-28.json)
在任何新 session 算法输出前固定：

- 独立单位为 capture session；
- Discovery/Development 为 ADVIO sequence13、14、15、17；
- sequence16 为 `SEALED_EVALUATION / SEALED_UNSEEN`；
- 每个 session 只使用 frames `0..601` 形成 601 个连续 pair，实际时长
  `10.015936–10.017468 s`；
- 每个 session 单进程、单一连续 `PairState`，只允许第一个 pair 为 baseline；
- R3 保持 strict `> 0.01/s`、三连续 pair、官方 `wxyz`、`T_cam_imu` optical
  basis、iPhone-03 去畸变有效区域掩膜和 `0.5` resize；
- 不汇总 pair 作为独立样本量，不计算 AUROC/F1。

sequence13、14、17 archive 合计 `437,001,516 bytes`，Zenodo MD5 与冻结值一致；
sequence15 复用既有 archive 和提取来源，并重新绑定三个输入 member 的 CRC、bytes 与
SHA-256。sequence16 archive 未取得。

## Session 级结果

表中 response 为每 session 内 evaluable pair 的 median absolute response；trigger
density 使用该 session 固定 601-pair 分母；Spearman 是各 session 内角速度与
absolute response 的描述性关联。pair 数不是推断样本量。

| session | support | raw / compensated abs response | raw / compensated trigger density | raw / compensated angular Spearman | failure |
| --- | ---: | ---: | ---: | ---: | --- |
| ADVIO 13 | 0.9900 | 0.06486 / 0.08484 | 0.5524 / 0.5291 | 0.6069 / 0.6140 | common-grid support 6 |
| ADVIO 14 | 0.9867 | 0.20897 / 0.21967 | 0.5008 / 0.5225 | 0.7316 / 0.7317 | common-grid support 8 |
| ADVIO 15 | 0.9967 | 0.05199 / 0.05194 | 0.3860 / 0.4043 | 0.3928 / 0.3920 | common-grid support 2 |
| ADVIO 17 | 0.9933 | 0.32948 / 0.32859 | 0.8769 / 0.8985 | 0.2054 / 0.2418 | common-grid support 4 |

高角速度层结果：

| session | cutoff deg/s | raw / compensated trigger density | raw / compensated abs response | disposition |
| --- | ---: | ---: | ---: | --- |
| ADVIO 13 | 18.0834 | 0.7355 / 0.8347 | 0.30805 / 0.32822 | `DETERIORATED` |
| ADVIO 14 | 19.9264 | 0.7273 / 0.7273 | 0.43763 / 0.43371 | `NOT_DETERIORATED` |
| ADVIO 15 | 27.4564 | 0.5620 / 0.7107 | 0.10629 / 0.12136 | `DETERIORATED` |
| ADVIO 17 | 29.9016 | 0.9421 / 1.0000 | 0.35062 / 0.35803 | `DETERIORATED` |

## 覆盖矩阵终态

固定片段没有事件级人工标签，因此不通过事后看图换片来补类别：

| 类别 | 终态 |
| --- | --- |
| 正常行走 | `DESCRIPTIVE_ONLY` |
| 转头 | `SOURCE_POSE_HIGH_ANGULAR_STRATUM` |
| 静态接近 | `NOT_EVALUABLE_NO_FROZEN_LABEL` |
| 横穿 | `NOT_EVALUABLE_NO_FROZEN_EVENT_LABEL` |
| 模糊 | `NOT_EVALUABLE_NO_FROZEN_BLUR_LABEL` |
| 步态振荡 | `DESCRIPTIVE_FAILURE_HYPOTHESIS_ONLY` |

这里没有把 `NOT_EVALUABLE` 类别替换为其他 clip，也没有从同一 session 切出多个
伪独立样本。

## 证据边界

结果只支持多 session 描述与机制路线停止。它不支持：

- RCLE 性能、泛化、AUROC/F1 或因果确认；
- 把 2,404 个 pair 写成 `n=2404`；
- sequence16 sealed evaluation 结论；
- reference-track、temporal consistency 或 bearing 的实现权限；
- Android、真人、产品、安全或生产结论。

机器结果位于
`artifacts.local/evidence/rcle_natural_session_expansion_discovery_r0/session_analysis_r0.json`，
SHA-256 为
`0174eb143d6f6d776345015478a858fa1ea03543422a7b8d1096f0ebe7dc8d5a`。
独立 validator 从四个 ledger 精确复算 session 结果、检查 sealed artifact 缺失与
禁用指标，验证收据与结果同目录保存；`validation_r0.json` 的 SHA-256 为
`9042aa0bec14a511e44cf1221be05a96667123db023f91f28fe78421db063b71`。
