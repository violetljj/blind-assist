# BlindAssist Assistive Geometry

状态：`current / R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / AG_R2_CROSS_SENSOR_CALIBRATION_CONTROL_R0_AND_R1_FAIL_CLOSED_CONSUMED / R1_INDEPENDENT_REPLAY_CONFIRMED_PRODUCER_FAILURE / SCIENTIFIC_NOT_RUN / CONFIRMATION_OUTCOMES_UNOPENED / DEFAULT_APP_UNCHANGED`

本页是路线日常操作真源。较早完整叙事保存在
[14d8ad7e 历史快照](archive/README_FULL_HISTORY_2026-08-13.md)，不能从中恢复旧权限。

## 当前主张

可替换轻量视觉 encoder 学习 metric-ish depth、support surface 与 obstacle boundary 连续因子，
再由确定性 body-swept reducer 构造 Clearance、Occupancy 与 `UNKNOWN`。DepthART-S 是可替换
encoder/initialization 候选，不是本路线的算法终点。

## 当前结论

- R2 F0 reducer 与 F1 supervision frontdoor 已通过；SuperTeacher → AG final V2 seam 已落地，
  但只证明接口与训练前门，不证明真实跨传感器精度。
- ETH3D calibration-control R0 与 R1 producer 均已 fail-closed consumed。R1 在读取 2 个 YAML /
  7,236 bytes 后以 `F2_R1_KALIBR_ROSTOPIC` 停止；matrix discovery 和 target-match count 保持
  `null/UNKNOWN`，没有 selected member/camera node。
- independent replay 又且仅又打开 archive 一次，复现同一 failure，并完成 producer/replay
  chain 验签；这证明失败可复现，不把它改写为科学 PASS/FAIL。
- session RGB-D/IMU、模型、truth、factor scoring 与 Confirmation outcome 均未访问；科学状态
  仍为 `NOT_RUN`，默认 App、Android/HTP、产品和安全权限不变。

## 当前证据入口

- [R2 factorized hypothesis](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)
- [F1 source-native supervision result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SOURCE_NATIVE_LABEL_MATERIALIZATION_AND_FRONTDOOR_RESULT_2026-08-11.json)
- [SuperTeacher → AG final V2 result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_SUPERTEACHER_TO_AG_LANDING_RESULT_2026-08-12.json)
- [Calibration-control R0 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_PREFLIGHT_ONE_SHOT_RESULT_2026-08-13.json)
- [R1 official camera-selection evidence](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R1_OFFICIAL_CAMERA_SELECTION_EVIDENCE_2026-08-13.json)
- [R1 terminal](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_CROSS_SENSOR_FACTOR_ACCURACY_CONFIRMATION_CALIBRATION_CONTROL_R1_ONE_SHOT_RESULT_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [研究脚本 Module](../../../scripts/research/assistive_geometry/README.md)

其余日期化 protocol/result/receipt 留在本目录；文件名新或包含 `LOCK/RESULT` 都不自动代表
current authority。

## 唯一 successor

无。R0/R1 producer 与 R1 independent replay 已消费。未来恢复须另立基于官方
archive-format evidence 的新版本、non-executing protocol 和 fresh root，并由用户单独授权。

## 当前允许

- 只读复核已签署 evidence、失败链和 frozen negative terminals；
- 在新版本 Development 中提出可证伪的新表示/机制，不改写已消费终态；
- 重放不访问真实 archive payload 的 synthetic/metadata focused tests。

## 当前禁止

- 重跑、resume、替换、覆盖或重新解释 R0/R1；
- 把 null discovery/match count 当 0 或 negative，或使用 first/best/camchain-order fallback；
- 访问 session RGB-D/IMU archive、运行模型/Confirmation 或用 reducer outcome 选 control；
- 把 control failure 写成 factor 科学 PASS/FAIL、默认 App、产品或安全证明。

## Claim ceiling

当前只证明 factor/reducer 与 supervision seam 的工程准备，以及 calibration-control failure
可复现；不证明真实跨传感器 factor accuracy、任务收益、部署可行性或助行安全。
