# BlindAssist 文档索引

本页是项目文档的稳定入口。`current` 表示当前操作或协议真源，`snapshot` 表示日期化结论，`archive` 表示历史材料。

## 开始使用与交付

| 文档 | 状态 | 用途 |
| --- | --- | --- |
| [NEW_COMPUTER_HANDOFF.md](NEW_COMPUTER_HANDOFF.md) | current | 新电脑与开发环境交接 |
| [RELEASE_AND_VERIFICATION.md](RELEASE_AND_VERIFICATION.md) | current | 发布和验证总流程 |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | current | 课程、阶段检查与答辩演示流程；演示前重新核验版本与设备状态 |
| [APK_ARCHIVE.md](APK_ARCHIVE.md) | current | APK 归档与 Git 里程碑策略 |
| [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md) | current | `artifacts.local/` 本地产物契约 |
| [CODEX_TASK_HANDOFF_TEMPLATE.md](CODEX_TASK_HANDOFF_TEMPLATE.md) | current | 多任务长任务的断点、边界与换窗口续作模板 |
| [CODEX_TASK_HANDOFF_INDEX_TEMPLATE.md](CODEX_TASK_HANDOFF_INDEX_TEMPLATE.md) | current | 多任务交接索引模板 |
| [DOCUMENT_GOVERNANCE.md](DOCUMENT_GOVERNANCE.md) | current | 文档职责、真源、历史保留与维护规则 |
| [RESEARCH_GOVERNANCE.md](RESEARCH_GOVERNANCE.md) | current | 分阶段最小证据包、科学/协议/权限三轴报告、薄修订、失败学习和反官僚约束 |
| [ENGINEERING_LEARNING_LOOP.md](ENGINEERING_LEARNING_LOOP.md) | current | 项目级异常识别、自我迭代、防复发与黑盒长任务规则 |
| [HOST_RESEARCH_COMPUTE.md](HOST_RESEARCH_COMPUTE.md) | current | 本机电脑端算法开发、CPU/GPU 并发、内存与长任务调度；不适用于 Android/边缘端 |
| [RESEARCH_PROTOCOL_TEMPLATE.md](RESEARCH_PROTOCOL_TEMPLATE.md) | current | LITE/STANDARD/STRICT 分级协议模板；机器合同默认留给高风险或确认性阶段 |
| [DEVICE_REGRESSION.md](DEVICE_REGRESSION.md) | current | 真机功能回归 |
| [AI_REVIEW_GOVERNANCE.md](AI_REVIEW_GOVERNANCE.md) | current | 端到端无人化：自主采集、标注、复核、裁决、准入、验收与发布证据总规则 |

## 产品、硬件与评测基线

| 文档 | 状态 | 用途 |
| --- | --- | --- |
| [GLASSES_HARDWARE_ROUTE.md](GLASSES_HARDWARE_ROUTE.md) | current | 眼镜硬件路线 |
| [BLINDASSIST_EVALSET.md](BLINDASSIST_EVALSET.md) | current | 助行图像评测集 |
| [DETECTOR_BENCHMARK.md](DETECTOR_BENCHMARK.md) | current | 检测器 benchmark 契约 |
| [NPU_DEFAULT_CANDIDATE.md](NPU_DEFAULT_CANDIDATE.md) | current | SM8650 QNN HTP 生产首选、CPU fallback 与新增 SoC 的 fail-closed 准入门 |
| [HETEROGENEOUS_PLATFORM_P0A_R0_2026-07-31.md](HETEROGENEOUS_PLATFORM_P0A_R0_2026-07-31.md) | snapshot | A568 历史目标的异构平台准入预检；终点为 `HOLD_NOT_EVALUABLE`，当前手机不作为 A568 替代 |
| [HETEROGENEOUS_PLATFORM_PHONE_P0A_R0_2026-07-31.md](HETEROGENEOUS_PLATFORM_PHONE_P0A_R0_2026-07-31.md) | snapshot | 当前连接 `SM-S9280 / SM8650` 手机准入预检；CPU/GPU 可进入 canary，QNN/HTP 保持 HOLD |
| [PROJECT_AUDIT_2026-07-28.md](PROJECT_AUDIT_2026-07-28.md) | snapshot | 全项目架构、研究治理、文档、脚本、构建与本地产物审查及优化 |
| [PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md](PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md) | snapshot | Project Guideline 八项组件的复用、适配、参考、暂缓与放弃边界 |
| [PROJECT_AUDIT_2026-07-10.md](PROJECT_AUDIT_2026-07-10.md) | snapshot | 2026-07-10 项目综合审计 |

## SANPO 当前协议与门禁

以下文件是 SANPO 路线的当前真源：

- [AI_REVIEW_GOVERNANCE.md](AI_REVIEW_GOVERNANCE.md)
- [SANPO_CURRENT_STATUS.md](SANPO_CURRENT_STATUS.md)
- [SANPO_TRAINING_PROTOCOL.md](SANPO_TRAINING_PROTOCOL.md)
- [SANPO_CANDIDATE_PROMOTION_GATES.md](SANPO_CANDIDATE_PROMOTION_GATES.md)
- [SANPO_SEQUENCE_EVALSET.md](SANPO_SEQUENCE_EVALSET.md)
- [SANPO_SEGMENTATION_CANDIDATE.md](SANPO_SEGMENTATION_CANDIDATE.md)
- [SANPO_TRAVERSABILITY_BASELINE.md](SANPO_TRAVERSABILITY_BASELINE.md)
- [SANPO_V3_REGRESSION_DATASET.md](SANPO_V3_REGRESSION_DATASET.md)
- [SANPO_V3_TEMPORARY_EXEMPTIONS.json](SANPO_V3_TEMPORARY_EXEMPTIONS.json)
- [SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md](SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md)
- [SANPO_P3_CONSENTED_CAPTURE_INTAKE_2026-07-13.md](SANPO_P3_CONSENTED_CAPTURE_INTAKE_2026-07-13.md)
- [SANPO_P3_VIEW_SOURCE_CONTRACT_2026-07-13.md](SANPO_P3_VIEW_SOURCE_CONTRACT_2026-07-13.md)
- [PUBLIC_VIDEO_GPT_SILVER_LABEL_PROTOCOL.md](PUBLIC_VIDEO_GPT_SILVER_LABEL_PROTOCOL.md)

## SANPO 日期化快照

- [SANPO_GPU_UTILIZATION_2026-07-13.md](SANPO_GPU_UTILIZATION_2026-07-13.md)
- [SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md](SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md)
- [SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md](SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md)
- [SANPO_P2_DETERMINISTIC_QUOTA_AUDIT_2026-07-13.md](SANPO_P2_DETERMINISTIC_QUOTA_AUDIT_2026-07-13.md)
- [SANPO_DETERMINISTIC_LINEAR_PROBE_2026-07-13.md](SANPO_DETERMINISTIC_LINEAR_PROBE_2026-07-13.md)
- [SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md](SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md)
- [SANPO_P3_OFFICIAL_TRAIN_FULL_DISCOVERY_2026-07-13.md](SANPO_P3_OFFICIAL_TRAIN_FULL_DISCOVERY_2026-07-13.md)
- [SANPO_EXTERNAL_EVENT_SOURCE_AUDIT_2026-07-15.md](SANPO_EXTERNAL_EVENT_SOURCE_AUDIT_2026-07-15.md)
- [SANPO_FEATURE_AND_DISTANCE_DIAGNOSTICS_2026-07-15.md](SANPO_FEATURE_AND_DISTANCE_DIAGNOSTICS_2026-07-15.md)
- [PUBLIC_SILVER_TRAINING_READINESS_2026-07-16.md](PUBLIC_SILVER_TRAINING_READINESS_2026-07-16.md)
- [PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md](PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md)
- [EXPLICIT_ROUTE_INTENT_MODEL_CONTRACT_2026-07-19.md](EXPLICIT_ROUTE_INTENT_MODEL_CONTRACT_2026-07-19.md)
- [SECONDARY_MODEL_TEST_PLAN_CORRIDOR_CAUSAL_STUDENT_2026-07-16.md](SECONDARY_MODEL_TEST_PLAN_CORRIDOR_CAUSAL_STUDENT_2026-07-16.md)
- [CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md](CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md)：独立候选的工程可行性、真值阻塞和禁止晋级结论。
- [ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md](ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)：`STOPPED / SUPERSEDED` 的历史 roadmap；当前终态以 [USTRF route-conditioned program 收口 R1](research/ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) 为准，YOLO/bbox 仅保留为普通 detector baseline。
- `SANPO_NEXT_LEAP_REPORT_SOURCE_2026-07-13.sql`：报告数据源，不是当前协议。

## 研究档案

- [Frontier Upgrade 2026-07 研究档案](research/frontier-upgrade-2026-07/README.md)：按主报告、来源证据、专题笔记、计划、任务包和审稿记录分层，只用于研究追溯。
- [研究文档媒体资产](research/assets/README.md)：研究文档引用的小型图表与图片；不存放数据集、模型输入或原始证据。

- [YOLO + 语义分割双环研究主线](research/dual-loop/README.md)：当前为
  `SEGMENTATION_MODEL_SELECTION_R1_BLOCKED / MODEL_SELECTION_NOT_EVALUABLE /
  THESIS_DEVELOPMENT_DEFAULT / DEVELOPMENT_REPAIR_AND_RERUN_ALLOWED /
  DEVELOPMENT_DEVICE_BENCHMARK_ALLOWED / NEXT_ROUTE_NOT_SELECTED /
  FINAL_CONFIRMATION_NOT_ACTIVATED / DEFAULT_APP_UNCHANGED`；历史 R1/R2-P0
  保持不可变，新 Development 可版本化修复重跑并提前采集工程 runtime，但不自动
  选择路线、接提醒或升级为最终确认。
  R2-P0 readiness 终态见
  [R2-P0 result](research/dual-loop/DUAL_LOOP_SEGMENTATION_R2_P0_RESULT_2026-08-01.md)，
  R1 consumed fresh 的永久角色修订见
  [consumed-role amendment](research/dual-loop/DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_CONSUMED_ROLE_AMENDMENT_2026-08-01.json)。
  中央图像阻塞前序结论保留为
  [D0-A successor R0 result](research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR_RESULT_2026-07-31.md)，
  [D0-A1 result](research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A1_RESULT_2026-07-31.md)，
  进入快照为
  [D0-A1 entry](research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A1_ENTRY_2026-07-31.md)，
  协议为
  [D0-A Agent 标签可用性协议](research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_PROTOCOL_2026-07-31.json)和
  [successor fixed-clip protocol](research/dual-loop/CENTRAL_OBSTRUCTION_AGENT_LABEL_READINESS_D0_A_SUCCESSOR_PROTOCOL_2026-07-31.json)。
  旧 active R1 只保留 Development mechanism、row-density diagnostic 与
  [事件失败分解](research/dual-loop/DUAL_LOOP_R1_EVENT_FAILURE_DECOMPOSITION_R0_RESULT_2026-07-31.md)，
  不授权默认生产、产品、安全或自动 R2。
- [RCLE 研究状态](research/rcle/README.md)：`paused / current truth`；保留 RCLE 的
  历史科学终态、协议终态、权限与禁止事项，不再是默认执行主线。
- [研究生组会进展总账](research/GROUP_MEETING_PROGRESS.md)：从项目建立至今的阶段工作、图文证据、统一研究问题、固定消融、统计计划与后续组会追加模板；不替代 SANPO/USTRF 当前状态真源。
- [USTRF-SC 研究记录与窗口交接](research/ustrf-sc/README.md)：历史 route-conditioned 研究和 RCLE 前序 Looming 证据；不再是研究主线，不代表生产授权。
- 开发日志月度历史：[2026-05](history/development-log/2026-05.md) · [2026-06](history/development-log/2026-06.md) · [2026-07](history/development-log/2026-07.md)。近期记录继续以根 `DEVELOPMENT_LOG.md` 为准。
- [早期项目计划、阶段与真机材料](history/project-materials/README.md)：archive；保留旧版本叙事，不证明当前状态。
- [想法池历史](history/idea/README.md)：archive；当前待决方向只在根 `idea.md` 维护。

## 维护规则

- 当前行为变化：更新对应 `current` 文档。
- 发布事实：写入根目录 `CHANGELOG.md`。
- 近期执行和验证：简洁写入根目录 `DEVELOPMENT_LOG.md`，详细证据链接到对应文档或 `artifacts.local/evidence/`。
- 一次性研究或审计：使用带日期文件，并明确标记为 snapshot。
- 新增顶层 `docs/*.md` 时同步更新本索引；运行 `scripts/check_docs_index.ps1`，避免只依赖文件名发现内容。
- 文档职责、当前真源和历史保留规则以 [DOCUMENT_GOVERNANCE.md](DOCUMENT_GOVERNANCE.md) 为准。
