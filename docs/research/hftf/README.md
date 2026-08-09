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
- DepthART-S 是当前研发主力候选：R0 为 `QUALITY_NOT_ADMITTED`，R1 保持 `RESEARCH_MAINLINE`；strict G4-D 为不可变负终态。Task-preserving D0 三臂已在 outcome 前技术前门关闭，没有 arm 进入任务质量或性能。D1 的最终 8-session × 300-frame Development roster、产品比例 `1×3×608×448` fixed-mixed 单候选与 fresh `SM-S9280 / SM8650 / HTP v75 / DZG1` 22,552,576-byte saved context 已冻结。用户已显式激活 D1 task-quality screen：2400 帧按 48 个 session 内固定 50-frame chunk 可恢复执行，首块 reference 与 50 个 HTP candidate 输出已通过 identity/hash/shape/finite 映射前门；全集 quality 聚合尚未完成，不能提前判 PASS/FAIL。首个 device attempt 因 `ADSP_LIBRARY_PATH` 分隔符错误在产生 candidate 输出前失败，受限 repair 只恢复已验证的 FastRPC 分号语义。PyTorch↔HTP raw-depth diagnostic 仍以 `max_abs=1.42328m` 明确 FAIL，只保留为 strict G4-D 负证据。R2 candidate 仍未选定；8 个 R2 session 继续 sealed。DA2 保持冻结 baseline/fallback。
- 既有 DA V2、FRESH-TF、Metric3D、ToF 和 temporal 结果保留为 Development、diagnostic 或 paused 证据，不能互相拼接成晋级结论。

## 稳定入口

- [DepthART R0 protocol/result](DEPTHART_ADMISSION_R0_PROTOCOL_2026-08-07.md) · [R0 result](DEPTHART_ADMISSION_R0_RESULT_2026-08-07.md)
- [DepthART R1 A3 result](DEPTHART_ADMISSION_R1_A3_RESULT_2026-08-07.md)
- [Task-preserving D0 precision screen](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.md) · [common source/control lock](DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json)
- [D0 FP16 technical preflight result](DEPTHART_TASK_PRESERVING_D0_FP16_TECHNICAL_PREFLIGHT_RESULT_2026-08-09.md)
- [D0 W8A16/INT8 shared calibration roster](DEPTHART_TASK_PRESERVING_D0_TUM_CALIBRATION_ROSTER_2026-08-09.json)
- [D0 precision screen terminal result](DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_RESULT_2026-08-09.md)
- [D1 fixed-mixed Development protocol](DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_PROTOCOL_2026-08-09.md) · [machine contract](DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_PROTOCOL_2026-08-09.json) · [metadata roster lock](DEPTHART_TASK_PRESERVING_D1_ARKIT_DEVELOPMENT_ROSTER_LOCK_2026-08-09.json)
- [D1 ARKitScenes scope receipt](DEPTHART_TASK_PRESERVING_D1_ARKIT_LICENSE_SCOPE_RECEIPT_2026-08-10.json) · [HEAD preflight protocol](DEPTHART_TASK_PRESERVING_D1_ARKIT_MEDIA_PREFLIGHT_PROTOCOL_2026-08-10.json) · [body preflight protocol](DEPTHART_TASK_PRESERVING_D1_ARKIT_BODY_PREFLIGHT_PROTOCOL_2026-08-10.json)
- [D1 label-blind body preflight result](DEPTHART_TASK_PRESERVING_D1_ARKIT_BODY_PREFLIGHT_RESULT_2026-08-10.md) · [machine result](DEPTHART_TASK_PRESERVING_D1_ARKIT_BODY_PREFLIGHT_RESULT_2026-08-10.json)
- [D1 product-aspect technical preflight](DEPTHART_TASK_PRESERVING_D1_PRODUCT_ASPECT_TECHNICAL_PREFLIGHT_RESULT_2026-08-10.md) · [candidate/reference/postprocess machine lock](DEPTHART_TASK_PRESERVING_D1_PRODUCT_ASPECT_TECHNICAL_PREFLIGHT_RESULT_2026-08-10.json)
- [D1 SM8650/v75 device protocol](DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_PROTOCOL_2026-08-10.json) · [device preflight result](DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_RESULT_2026-08-10.md) · [machine result](DEPTHART_TASK_PRESERVING_D1_SM8650_HTP_CONTEXT_AND_OUTCOME_ACTIVATION_PREFLIGHT_RESULT_2026-08-10.json)
- [D1 quality-screen protocol](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_PROTOCOL_2026-08-10.json) · [activation receipt](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_ACTIVATION_2026-08-10.json) · [bounded runner repair](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_RUNNER_REPAIR_2026-08-10.json)
- [DepthART task-preserving deployment R2 protocol](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md) · [machine contract](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json)
- [R2 ARKit roster lock](DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json) · [media HEAD preflight](DEPTHART_TASK_PRESERVING_R2_ARKIT_MEDIA_PREFLIGHT_2026-08-09.json)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)

## 唯一 successor

`COMPLETE_FROZEN_D1_DEVELOPMENT_TASK_QUALITY_SCREEN`：activation receipt 已生效，唯一下一步是按冻结
48-chunk 顺序完成余下 Development reference/candidate outputs，再一次性计算 clearance、false-clear、
false-block、temporal 与 geometry gates。不得用部分块作选择或结论，不得修改 candidate、数据、后处理、
known-coverage 分母或门限。任一 required aggregate/denominator/stratum 缺失即 fail；quality PASS 之前不得
测性能，R2 cohort 继续禁止访问。

R2 是下游 sealed 路线，不是并列 successor：metadata roster 已冻结但 candidate 尚未选定；
D1 即使 PASS 也只建立一个 R2 candidate lock，不能直接访问独立 outcome，也不产生 scientific
admission、DA2 替换或默认 App 权限。

近完整 custom-float32 engine 与新 runtime/hardware 保留为未激活的新立项候选，不能从
R2 或旧 G4-D 自动获得执行权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、
把 HTP/Android canary 接入默认 App、或将任何 DepthART 结果写成生产权限。没有满足
successor 条件时，路线保持 `diagnostic` 或 `paused`。
