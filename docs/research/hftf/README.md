# DepthART 算法路线

状态：`current / DEVELOPMENT_STANDARD / INNOVATION_NOT_EVALUABLE / R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / D1_FINAL_8_SESSION_ROSTER_LOCKED / D1_608X448_SM8650_V75_CONTEXT_AND_EXECUTION_PREFLIGHT_PASS / RAW_DEPTH_PARITY_DIAGNOSTIC_FAIL / D1_QUALITY_SCREEN_48_OF_48_COMPLETE / D1_TASK_QUALITY_FAIL_TERMINAL / D2_MECHANICS_CANARY_PASS / D2R1_SOURCE_SUPPORT_PASS_16_OF_16 / D2_4_TRAIN_4_DEVELOPMENT_SEALED / D2_PHASE_C_RGB_HEAD_PASS_8_OF_8 / D2_PHASE_C_BODY_AWAITING_EXPLICIT_SCOPE / D2_MODEL_AND_TRAINING_NOT_ACTIVATED / R2_CANDIDATE_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

本页只维护当前摘要、权限和唯一 successor。完整历史已保留在 [archive/README_FULL_HISTORY_2026-08-07.md](archive/README_FULL_HISTORY_2026-08-07.md)，日期化协议、receipt 和结果仍是 snapshot/机器证据。

## 当前主张

DepthART-S 当前是 [BlindAssist Assistive Geometry](../assistive-geometry/README.md) 的优先
encoder/initialization、depth baseline 与部署研究载体，不是算法终点。本页只维护 DepthART 路线；项目级数据、
通信链路、端到端延迟、性能和部署研究分别从 [研究总入口](../README.md) 进入，只有显式绑定时
才成为本路线的输入或证据。不等于正式 App 能力。

## 当前状态

- DepthART 算法路线与双环论文次线隔离，默认 App 和正式 YOLO 模型不变。
- DA2 保持冻结的 metric teacher、baseline、regression reference 和 fallback，不因新候选结果删除或降级。
- DepthART-S 是当前研发主力候选：R0 为 `QUALITY_NOT_ADMITTED`，R1 保持 `RESEARCH_MAINLINE`；strict G4-D 为不可变负终态。Task-preserving D0 三臂已在 outcome 前技术前门关闭。D1 已在冻结的 8-session × 300-frame Development roster、产品比例 `1×3×608×448` fixed-mixed 单候选与 fresh `SM-S9280 / SM8650 / HTP v75 / DZG1` saved context 上完成 48/48 个 device chunk、2400 帧和 21600 cells 的一次性汇总。终态为 `D1_TASK_QUALITY_FAIL_STOP_R2_CANDIDATE_NOT_AUTHORIZED`：候选 pooled clearance MAE `0.38443 m`、false-clear `0.16651`、false-block `0.18648`、geometry transition agreement `0.79365`，均未满足对应绝对门；false-block noninferiority 也失败。另有 required parent aggregates 非 finite，按协议 fail-closed。R2 candidate 不授权，8 个 R2 session 继续 sealed；性能、DA2 替换、默认 App、production 与 safety 均未授权。
- D2 只把 D1 失败结构转成一个小型 task-evidence head 假设和新的 source-support 路线，没有改写 D1 outcome。277 参数 head 的 synthetic canary 仅通过 shape、UNKNOWN、horizon monotonicity 与 bounded-residual mechanics。Phase-A 锁定 16 个连续 portrait/pose identity；固定首窗口的 Phase-B 仅 2 个通过，但 D2R1 在同一 16 个身份的全部连续 portrait runs 中以未修改门限扫描，最终 16/16 均找到合格 300-frame window。冻结顺序的前 4 个已锁为 D2 TRAIN，后 4 个已锁为 sealed D2 DEVELOPMENT；没有读取 RGB、运行模型、训练或打开 Development outcome。
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
- [D1 quality-screen protocol](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_PROTOCOL_2026-08-10.json) · [activation receipt](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_ACTIVATION_2026-08-10.json) · [bounded runner repair](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_RUNNER_REPAIR_2026-08-10.json) · [terminal result](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_RESULT_2026-08-11.md) · [machine summary](DEPTHART_TASK_PRESERVING_D1_QUALITY_SCREEN_RESULT_2026-08-11.json)
- [D2 task-evidence head protocol](DEPTHART_TASK_PRESERVING_D2_TASK_EVIDENCE_HEAD_PROTOCOL_2026-08-11.md) · [source-support pool lock](DEPTHART_TASK_PRESERVING_D2_SOURCE_SUPPORT_POOL_LOCK_2026-08-11.json) · [scoped source receipt](DEPTHART_TASK_PRESERVING_D2_ARKIT_SOURCE_SCOPE_RECEIPT_2026-08-11.json)
- [D2 Phase-A result](DEPTHART_TASK_PRESERVING_D2_PHASE_A_RESULT_2026-08-11.json) · [Phase-B terminal result](DEPTHART_TASK_PRESERVING_D2_PHASE_B_RESULT_2026-08-11.md) · [partial-role repair](DEPTHART_TASK_PRESERVING_D2_PHASE_B_PARTIAL_ROLE_REPAIR_2026-08-11.json)
- [D2R1 target-support window recovery protocol](DEPTHART_TASK_PRESERVING_D2R1_TARGET_SUPPORT_WINDOW_RECOVERY_PROTOCOL_2026-08-11.md)
- [D2R1 governed result](DEPTHART_TASK_PRESERVING_D2R1_RESULT_2026-08-11.md) · [machine result](DEPTHART_TASK_PRESERVING_D2R1_RESULT_2026-08-11.json) · [Phase-C RGB HEAD scope](DEPTHART_TASK_PRESERVING_D2_PHASE_C_RGB_HEAD_SCOPE_PROTOCOL_2026-08-11.md)
- [D2 Phase-C RGB HEAD result](DEPTHART_TASK_PRESERVING_D2_PHASE_C_RGB_HEAD_RESULT_2026-08-11.md) · [exact-eight body materialization scope](DEPTHART_TASK_PRESERVING_D2_PHASE_C_BODY_MATERIALIZATION_SCOPE_PROTOCOL_2026-08-11.md)
- [DepthART task-preserving deployment R2 protocol](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md) · [machine contract](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json)
- [R2 ARKit roster lock](DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json) · [media HEAD preflight](DEPTHART_TASK_PRESERVING_R2_ARKIT_MEDIA_PREFLIGHT_2026-08-09.json)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)

## 唯一 successor

唯一 successor 是 `EXPLICIT_D2_PHASE_C_EXACT_EIGHT_BODY_MATERIALIZATION_ONLY`，当前
`execution=false`。只有用户显式授权后，才允许对 D2R1 exact 4 TRAIN + 4 sealed DEVELOPMENT
identity 请求 32 个 RGB/intrinsics/depth/confidence ZIP body，精确总量 `5,281,655,713` bytes、
上限 `5,282,000,000` bytes，并按已锁 stems 机械提取各 300 帧。不得解码图像、重新选择窗口、
派生 truth、运行模型、训练或打开 Development；RGB HEAD 权限不能自动继承为 GET。

R2 是下游 sealed 路线，不是并列 successor：metadata roster 已冻结但 candidate 尚未选定；
D2R1 source-support PASS 只建立 D2 数据角色；不能直接访问 R2 outcome，也不产生 scientific
admission、DA2 替换或默认 App 权限。

近完整 custom-float32 engine 与新 runtime/hardware 保留为未激活的新立项候选，不能从
R2 或旧 G4-D 自动获得执行权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、
把 HTP/Android canary 接入默认 App、或将任何 DepthART 结果写成生产权限。没有满足
successor 条件时，路线保持 `diagnostic` 或 `paused`。
