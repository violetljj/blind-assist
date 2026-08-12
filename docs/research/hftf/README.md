# DepthART 算法路线

状态：`current / DEVELOPMENT_STANDARD / INNOVATION_NOT_EVALUABLE / R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D0_NO_ELIGIBLE_PRECISION_ARM / D1_TASK_QUALITY_FAIL_TERMINAL / D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_TERMINAL / D3_BIDIRECTIONAL_ROUTER_MECHANICS_PASS / D3_FRESH_METADATA_POOL_48_CONSUMED / D3_PHASE_A_HEAD_96_OF_96_PASS / D3_PHASE_A_FAIL_21_OF_32_NO_SELECTION / D3R1_RECOVERY_PROTOCOL_FROZEN / D3R1_PHASE_A_BODY_127_PROCESSED_53_ELIGIBLE_32_LOCKED / D3R1_PHASE_B_EXECUTION_INVALID_INCOMPLETE_NO_SCIENTIFIC_TERMINAL / R2_CANDIDATE_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

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
- D2 只把 D1 失败结构转成一个小型 task-evidence head 假设和新的 source-support 路线，没有改写 D1 outcome。TRAIN-only 已完成 24/24 chunks 并锁定 step-500 / 277 参数 head。冻结的 4-identity × 300-frame Development screen 随后完成 24/24 chunks、1,200 帧与 10,800 cells，终态为 `D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_STOP`。head 将 pooled clearance MAE 从 `0.43615 m` 降至 `0.27931 m`、false-clear 从 `0.20795` 降至 `0.08599`，但仍未满足绝对门；false-block 升至 `0.37614`，known-coverage/false-block/valid-to-unknown noninferiority 失败，并有 required strata denominator 为空。D2 outcome 已消费，不允许事后修改 checkpoint、阈值、数据、postprocess、denominator 或 gate 回救。
- D3 已另立为新的 bidirectional error-certificate router，不重跑 D2 direct-state head。它分别学习 `CLEAR release` 与 `OCCUPIED veto`，证据冲突时转 `UNKNOWN_GROUND`，证据不足时保持 baseline；目标是在 fresh identity-disjoint Development 上同时降低 false-clear 与 false-block。纯 CPU mechanics canary 已冻结并 PASS，metadata planner 排除 394 个既有官方 identity 后锁定 48 个全新 Training-fold 候选，与 D1、D2、sealed R2 overlap 均为 0。Phase-A 已完整下载并独立复核 exact 96 个 intrinsics/trajectory body（`41,979,912` bytes、`190,028` 个 `.pincam` payload），但固定 300-frame portrait/pose continuity 仅 `21/48` identity 通过，少于所需 32。终态为 `D3_PHASE_A_FAIL_FEWER_THAN_32_ELIGIBLE_IDENTITIES`；正式 selection lock、TRAIN/DEVELOPMENT role 和 Phase-B authority 均为空，当前 D3 evidence version 关闭。
- D3R1 是独立 pre-outcome recovery version，不扩大或复用旧 D3 exact-48/合格 21，也不降低 continuity 门。它以旧版 `21/48` 的单侧 95% Clopper-Pearson 下界仅作资源规划，冻结最小 planning pool `127`；该 IID heuristic 不是科学预测、质量门或通过保证。metadata planner 从 published pre-recovery commit 的 immutable `docs/research` tree 排除 490 个官方 token，并额外 SHA 锚定 TARO R10 scripts-only fresh pool 的 64 个 token；effective firewall 554，锁定 127 个 unique visit/session，所有显式 overlap 为 0。Phase-A exact-127 intrinsics/trajectory HEAD 与 body 已完成：254 bodies 共 `133,734,849` bytes，完整处理 127 identities，校验 `603,634` 个 `.pincam` 与 `99,155` 个 trajectory rows；producer 与独立离线 validator 一致得到 `53/127` continuity eligible，并按 frozen pool order 锁定最早 32 个。Phase-B 为这 exact-32 与绑定的 9,600 个 frame stems 登记并完成 64/64 depth/confidence HEAD PASS；随后 body 执行在第 `2/32` 身份、checkpoint 002 前发现冻结 stem `42898216_694900.389` 不在 source depth inventory，按 exact-coverage hard gate 停止。当前 r0 只有 1 个完整 checkpoint 和 2 个身份的 4 个 source bodies；没有 scientific manifest/validation、没有 selection 或 first-16 lock。此为 `INVALID_INCOMPLETE`，不是科学 PASS/FAIL 或 `D3_DATA_SUPPORT_NOT_EVALUABLE`；当前 r0 不可 resume、repair、覆盖、换帧、换身份或重跑，scientific terminal 与 successor 均为 null。RGB、模型、角色、训练、Development、R2 继续未打开。
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
- [D2 Phase-C source result](DEPTHART_TASK_PRESERVING_D2_PHASE_C_SOURCE_RESULT_2026-08-11.md) · [TRAIN-only activation scope](DEPTHART_TASK_PRESERVING_D2_TRAIN_ONLY_ACTIVATION_SCOPE_PROTOCOL_2026-08-11.md)
- [D2 TRAIN-only governed result](DEPTHART_TASK_PRESERVING_D2_TRAIN_ONLY_RESULT_2026-08-11.md) · [Development quality activation scope](DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_QUALITY_ACTIVATION_SCOPE_PROTOCOL_2026-08-11.md) · [Development quality terminal](DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_QUALITY_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D2_DEVELOPMENT_QUALITY_RESULT_2026-08-12.json)
- [D3 bidirectional error-certificate router protocol](DEPTHART_TASK_PRESERVING_D3_BIDIRECTIONAL_ERROR_CERTIFICATE_ROUTER_PROTOCOL_2026-08-12.md) · [machine contract](DEPTHART_TASK_PRESERVING_D3_BIDIRECTIONAL_ERROR_CERTIFICATE_ROUTER_PROTOCOL_2026-08-12.json) · [mechanics result](DEPTHART_TASK_PRESERVING_D3_BIDIRECTIONAL_ROUTER_MECHANICS_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D3_BIDIRECTIONAL_ROUTER_MECHANICS_RESULT_2026-08-12.json)
- [D3 fresh metadata roster](DEPTHART_TASK_PRESERVING_D3_FRESH_METADATA_ROSTER_LOCK_2026-08-12.json) · [metadata result](DEPTHART_TASK_PRESERVING_D3_FRESH_METADATA_ROSTER_RESULT_2026-08-12.md) · [source-scope receipt](DEPTHART_TASK_PRESERVING_D3_SOURCE_SCOPE_AND_METADATA_ROSTER_RECEIPT_2026-08-12.json)
- [D3 Phase-A HEAD protocol](DEPTHART_TASK_PRESERVING_D3_PHASE_A_HEAD_PROTOCOL_2026-08-12.json) · [activation](DEPTHART_TASK_PRESERVING_D3_PHASE_A_HEAD_ACTIVATION_2026-08-12.json) · [HEAD result](DEPTHART_TASK_PRESERVING_D3_PHASE_A_HEAD_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D3_PHASE_A_HEAD_RESULT_2026-08-12.json)
- [D3 Phase-A body protocol](DEPTHART_TASK_PRESERVING_D3_PHASE_A_BODY_PROTOCOL_2026-08-12.json) · [activation](DEPTHART_TASK_PRESERVING_D3_PHASE_A_BODY_ACTIVATION_2026-08-12.json) · [validator repair](DEPTHART_TASK_PRESERVING_D3_PHASE_A_TERMINAL_VALIDATOR_REPAIR_2026-08-12.json) · [terminal result](DEPTHART_TASK_PRESERVING_D3_PHASE_A_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D3_PHASE_A_RESULT_2026-08-12.json)
- [D3R1 Phase-A recovery protocol](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_RECOVERY_PROTOCOL_2026-08-12.md) · [machine protocol](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_RECOVERY_PROTOCOL_2026-08-12.json) · [metadata activation](DEPTHART_TASK_PRESERVING_D3R1_METADATA_ACTIVATION_2026-08-12.json) · [fresh roster lock](DEPTHART_TASK_PRESERVING_D3R1_FRESH_METADATA_ROSTER_LOCK_2026-08-12.json) · [result](DEPTHART_TASK_PRESERVING_D3R1_FRESH_METADATA_ROSTER_RESULT_2026-08-12.md) · [source-scope receipt](DEPTHART_TASK_PRESERVING_D3R1_SOURCE_SCOPE_RECEIPT_2026-08-12.json)
- [D3R1 Phase-A HEAD protocol](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_HEAD_PROTOCOL_2026-08-12.json) · [activation](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_HEAD_ACTIVATION_2026-08-12.json) · [HEAD result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_HEAD_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_HEAD_RESULT_2026-08-12.json)
- [D3R1 Phase-A body protocol](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_BODY_PROTOCOL_2026-08-12.json) · [activation](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_BODY_ACTIVATION_2026-08-12.json) · [result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_RESULT_2026-08-12.json) · [Phase-B depth/confidence source-scope receipt](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_DEPTH_CONFIDENCE_SOURCE_SCOPE_RECEIPT_2026-08-12.json)
- [D3R1 Phase-B HEAD protocol](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_HEAD_PROTOCOL_2026-08-12.json) · [activation](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_HEAD_ACTIVATION_2026-08-12.json) · [result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_HEAD_RESULT_2026-08-12.md) · [machine result](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_HEAD_RESULT_2026-08-12.json)
- [D3R1 Phase-B body protocol](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_BODY_PROTOCOL_2026-08-12.json) · [activation](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_BODY_ACTIVATION_2026-08-12.json) · [execution stop](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_EXECUTION_STOP_2026-08-12.md) · [machine receipt](DEPTHART_TASK_PRESERVING_D3R1_PHASE_B_EXECUTION_STOP_2026-08-12.json)
- [DepthART task-preserving deployment R2 protocol](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.md) · [machine contract](DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2_PROTOCOL_2026-08-09.json)
- [R2 ARKit roster lock](DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json) · [media HEAD preflight](DEPTHART_TASK_PRESERVING_R2_ARKIT_MEDIA_PREFLIGHT_2026-08-09.json)
- [DA2 P1/P2 closure](DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md)
- [HFTF candidate charter](HFTF_CANDIDATE_LANE_CHARTER_R0_2026-08-01.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)

## 唯一 successor

`NONE`。D3R1 Phase-B r0 已在 exact-frame source coverage hard gate 停止，
`scientific_terminal=null / successor=null`。不得删除 orphan source 后强行 resume，不得用邻帧、其余
Phase-A eligible identity 或同版本新 root 回救。任何未来恢复都必须由用户另行授权并另立新版本、
协议和 root；它不是当前路线的自动 successor，也不得受已打开的 partial source-support 结果调节。
旧 D3 exact-48/21-qualified 版本仍保持 fail terminal，不能 carry-over 或扩池回救。

R2 是下游 sealed 路线，不是并列 successor：metadata roster 已冻结但 candidate 尚未选定；
D2R1 source-support PASS 只建立 D2 数据角色；不能直接访问 R2 outcome，也不产生 scientific
admission、DA2 替换或默认 App 权限。

近完整 custom-float32 engine 与新 runtime/hardware 保留为未激活的新立项候选，不能从
R2 或旧 G4-D 自动获得执行权限。

## 禁止与权限边界

禁止在 consumed 数据上调参回救、把 teacher/合成/单设备性能当作 accuracy 或 safety、
把 HTP/Android canary 接入默认 App、或将任何 DepthART 结果写成生产权限。没有满足
successor 条件时，路线保持 `diagnostic` 或 `paused`。
