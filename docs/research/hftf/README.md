# DepthART 算法路线

状态：`current / DEVELOPMENT_STANDARD / INNOVATION_NOT_EVALUABLE / DEFAULT_APP_UNCHANGED`

本页只维护当前摘要、权限和唯一 successor。完整历史已保留在 [archive/README_FULL_HISTORY_2026-08-07.md](archive/README_FULL_HISTORY_2026-08-07.md)，日期化协议、receipt 和结果仍是 snapshot/机器证据。

## 当前主张

DepthART-S 当前是 [BlindAssist Assistive Geometry](../assistive-geometry/README.md) 的优先
encoder/initialization、depth baseline 与部署研究载体，不是算法终点。本页只维护 DepthART 路线；项目级数据、
通信链路、端到端延迟、性能和部署研究分别从 [研究总入口](../README.md) 进入，只有显式绑定时
才成为本路线的输入或证据。不等于正式 App 能力。

## 当前状态

- DepthART 算法路线与双环论文次线隔离，默认 App 和正式 YOLO 模型不变。
- DA2 保持冻结的 metric teacher、baseline、regression reference 和 fallback，不因新候选结果删除或降级。
- DepthART-S 是当前研发主力候选：R0 为 `QUALITY_NOT_ADMITTED`，R1 保持 `RESEARCH_MAINLINE`；strict G4-D 为不可变负终态。Task-preserving D0 三臂已在 outcome 前技术前门关闭：FP16 的 custom islands 漂移为不兼容 Float16；W8A16/INT8 的 QAIRT Windows host quantizer 缺少 SelectiveScan CPU package，无法构图。没有 arm 进入任务质量或性能，R2 candidate 仍未选定。既有 G4-C fixed-mixed 图仍只是 `448×448` 数值 canary source/control；若继续，必须另立 D1 并先冻结产品纵横比输入、FOV/resize、intrinsics/truth 对齐、task postprocess 与新 Development roster，不能事后塞回 D0。8 个 ARKitScenes R2 session 仍 metadata-only 且禁止参与选模。DA2 保持冻结 baseline/fallback。
- 既有 DA V2、FRESH-TF、Metric3D、ToF 和 temporal 结果保留为 Development、diagnostic 或 paused 证据，不能互相拼接成晋级结论。

## 稳定入口

- [DepthART R0 protocol/result](DEPTHART_ADMISSION_R0_PROTOCOL_2026-08-07.md) · [R0 result](DEPTHART_ADMISSION_R0_RESULT_2026-08-07.md)
- [DepthART R1 A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)
- [Task-preserving D0 precision screen](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.md) · [common source/control lock](DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json)
- [D0 FP16 technical preflight result](DEPTHART_TASK_PRESERVING_D0_FP16_TECHNICAL_PREFLIGHT_RESULT_2026-08-09.md)
- [D0 W8A16/INT8 shared calibration roster](DEPTHART_TASK_PRESERVING_D0_TUM_CALIBRATION_ROSTER_2026-08-09.json)
- [D0 precision screen terminal result](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.md)
- [DepthART task-preserving deployment R2 protocol](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md) · [machine contract](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json)
- [R2 ARKit roster lock](DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json) · [media HEAD preflight](DEPTHART_TASK_PRESERVING_R2_ARKIT_MEDIA_PREFLIGHT_2026-08-09.json)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)

## 唯一 successor

`DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN`：D0 三条 recipe 已无合格 arm，
不得修改后重跑。下一步只允许把 fixed-mixed 路线作为新问题处理：先冻结产品纵横比输入、
FOV/resize、intrinsics 与 truth 对齐、depth→ground→clearance→risk 后处理及新 Development
roster，再重建对应图并做单候选任务 screen。
它通过也只能建立一个 R2 candidate lock，不能直接访问独立 outcome。

`DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2`：strict G4-D 保持不可变负终态。R2 metadata roster
已冻结但 candidate 尚未选定；R2 cohort 不得用于 calibration 或三臂选择。只有 D0 选出并锁定
单一候选、license scope 明确扩展、cohort integrity preflight 和显式 activation 完成后，才可
访问 R2 task outcome。部署结果仍不产生 scientific admission、DA2 替换或默认 App 权限。

近完整 custom-float32 engine 与新 runtime/hardware 保留为未激活的新立项候选，不能从
R2 或旧 G4-D 自动获得执行权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、
把 HTP/Android canary 接入默认 App、或将任何 DepthART 结果写成生产权限。没有满足
successor 条件时，路线保持 `diagnostic` 或 `paused`。
