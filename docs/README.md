# BlindAssist 文档中心

本页是面向人的稳定入口。`current` 是当前操作/协议真源，`snapshot` 是日期化结论，
`archive` 是历史材料；后两者都不自动产生当前权限。

## 先选身份

| 你要做什么 | 最短入口 | 默认不要先读 |
|---|---|---|
| 安装、体验或了解项目 | [中文快速开始](QUICKSTART_ZH.md) · [English quick start](QUICKSTART_EN.md) · [演示指南](DEMO_GUIDE.md) | 研究 archive、完整开发日志 |
| 贡献代码或文档 | [CONTRIBUTING](../CONTRIBUTING.md) · [代码地图](CODE_MAP.md) · [文档治理](DOCUMENT_GOVERNANCE.md) | 无关研究路线、设备/发布合同 |
| 开展或复核研究 | [研究总入口](research/README.md) · [研究治理](RESEARCH_GOVERNANCE.md) | 按文件日期猜 current、一次读遍所有路线 |
| 维护、交付或换任务 | [项目冷启动](PROJECT_STATE.md) · [Codex workflow](CODEX_WORKFLOW.md) · [发布验证](RELEASE_AND_VERIFICATION.md) | 旧 handoff、snapshot、全量历史 |

## 用户、演示与开源

| 文档 | 状态 | 用途 |
|---|---|---|
| [QUICKSTART_ZH.md](QUICKSTART_ZH.md) · [QUICKSTART_EN.md](QUICKSTART_EN.md) | current | 中英文三分钟开始路径 |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | current | 课程、阶段检查与答辩演示；演示前重新核验设备和版本 |
| [OPEN_SOURCE_PUBLIC_VALUE.md](OPEN_SOURCE_PUBLIC_VALUE.md) | current | 公共价值、受益对象、证据边界和维护资源使用计划 |
| [COMMUNITY_LAUNCH_KIT.md](COMMUNITY_LAUNCH_KIT.md) | current | 发布叙事、真实设备演示合同、贡献者招募和渠道文案 |
| [MODEL_CARD.md](MODEL_CARD.md) | current | 默认 App 模型身份、来源、许可证、用途和限制 |
| [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) | current | 根许可证与第三方模型、依赖、数据和媒体边界 |
| [THREAT_MODEL.md](THREAT_MODEL.md) | current | Android、模型、CI、发布、数据和 AI 维护的威胁边界 |

## 贡献者与维护者

| 文档 | 状态 | 用途 |
|---|---|---|
| [PROJECT_STATE.md](PROJECT_STATE.md) | current | 维护者/Agent 的 30 秒冷启动和最小读取路由 |
| [CODE_MAP.md](CODE_MAP.md) | current | 稳定模块职责、实现入口和默认跳过路径 |
| [NEW_COMPUTER_HANDOFF.md](NEW_COMPUTER_HANDOFF.md) | current | 新电脑和开发环境交接 |
| [CODEX_WORKFLOW.md](CODEX_WORKFLOW.md) | current | 任务范围、共享工作树、最小验证和直接交付规则 |
| [CODEX_MAINTAINER_AUTOMATION.md](CODEX_MAINTAINER_AUTOMATION.md) | current | Codex/API 开源维护输入、输出、权限和密钥边界 |
| [CODEX_TASK_HANDOFF_TEMPLATE.md](CODEX_TASK_HANDOFF_TEMPLATE.md) · [CODEX_TASK_HANDOFF_INDEX_TEMPLATE.md](CODEX_TASK_HANDOFF_INDEX_TEMPLATE.md) | current | 真正需要跨窗口时的本地断点模板 |
| [DOCUMENT_GOVERNANCE.md](DOCUMENT_GOVERNANCE.md) | current | 文档职责、唯一真源、操作面预算和历史保留 |
| [ENGINEERING_LEARNING_LOOP.md](ENGINEERING_LEARNING_LOOP.md) | current | 异常识别、自我迭代、防复发和长任务规则 |
| [LOCAL_ARTIFACTS.md](LOCAL_ARTIFACTS.md) | current | `artifacts.local/` 本地产物边界 |
| [HOST_RESEARCH_COMPUTE.md](HOST_RESEARCH_COMPUTE.md) | current | 电脑端 CPU/GPU、内存和长任务调度 |
| [DEVICE_REGRESSION.md](DEVICE_REGRESSION.md) | current | 真机功能、性能和无障碍回归矩阵 |
| [RELEASE_AND_VERIFICATION.md](RELEASE_AND_VERIFICATION.md) · [APK_ARCHIVE.md](APK_ARCHIVE.md) | current | 发布验证和 APK 归档 |
| [AI_REVIEW_GOVERNANCE.md](AI_REVIEW_GOVERNANCE.md) | current | 自主采集、标注、复核、准入和发布证据边界 |
| [RESEARCH_PROTOCOL_TEMPLATE.md](RESEARCH_PROTOCOL_TEMPLATE.md) | current | LITE/STANDARD/STRICT 分级协议模板 |

## 产品、硬件、评测与 SANPO

| 文档 | 状态 | 用途 |
|---|---|---|
| [GLASSES_HARDWARE_ROUTE.md](GLASSES_HARDWARE_ROUTE.md) | current | 眼镜硬件路线 |
| [BLINDASSIST_EVALSET.md](BLINDASSIST_EVALSET.md) · [DETECTOR_BENCHMARK.md](DETECTOR_BENCHMARK.md) | current | 助行评测集和 detector benchmark 契约 |
| [NPU_DEFAULT_CANDIDATE.md](NPU_DEFAULT_CANDIDATE.md) | current | SM8650/QNN HTP 候选、CPU fallback 和新 SoC 准入 |
| [SANPO_CURRENT_STATUS.md](SANPO_CURRENT_STATUS.md) | current | SANPO 当前状态、双通道和禁止事项唯一摘要 |
| [SANPO_TRAINING_PROTOCOL.md](SANPO_TRAINING_PROTOCOL.md) · [SANPO_CANDIDATE_PROMOTION_GATES.md](SANPO_CANDIDATE_PROMOTION_GATES.md) | current | Development 训练与显式 production promotion 分层 |
| [SANPO_SEQUENCE_EVALSET.md](SANPO_SEQUENCE_EVALSET.md) · [SANPO_V3_REGRESSION_DATASET.md](SANPO_V3_REGRESSION_DATASET.md) | current | 连续序列与回归数据合同 |
| [SANPO_SEGMENTATION_CANDIDATE.md](SANPO_SEGMENTATION_CANDIDATE.md) · [SANPO_TRAVERSABILITY_BASELINE.md](SANPO_TRAVERSABILITY_BASELINE.md) | current | 分割候选和可通行性 baseline |
| [SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md](SANPO_COUNTERFACTUAL_EPISODE_COLLECTION.md) | current | 反事实 episode 采集合同 |
| [PUBLIC_VIDEO_GPT_SILVER_LABEL_PROTOCOL.md](PUBLIC_VIDEO_GPT_SILVER_LABEL_PROTOCOL.md) | current | public-video silver-label 边界，不等于 source GT |

## 研究入口

从[研究总入口](research/README.md)选择算法、数据或系统/平台分类。下表只定位领域，
不复制动态状态、指标或 successor。

| 类型 | 入口 |
|---|---|
| 分类 current | [算法](research/ALGORITHM_RESEARCH_CURRENT.md) · [数据](research/DATA_RESEARCH_CURRENT.md) · [系统与平台](research/SYSTEM_RESEARCH_CURRENT.md) |
| 全局失败清算 | [Failure Synthesis](BLINDASSIST_FAILURE_SYNTHESIS.md) · [Causal Failure Model](BLINDASSIST_CAUSAL_FAILURE_MODEL.md) · [Oracle Ladder](BLINDASSIST_ORACLE_LADDER.md) |
| 因果诊断与算法路线 | [D-ORACLE-1](research/failure-synthesis/README.md) · [SVRF](research/svrf/README.md) · [VI-Task Geometry](research/vi-task-geometry/README.md) · [GA-SATOM](research/ga-satom/README.md) · [SATOM-A](research/satom/README.md) · [Assistive Geometry](research/assistive-geometry/README.md) · [TARO](research/taro/README.md) · [AG-QSF](research/assistive-geometry-qsf/README.md) · [AG-CBF](research/assistive-geometry-cbf/README.md) · [DepthART](research/hftf/README.md) · [双环](research/dual-loop/README.md) · [RCLE](research/rcle/README.md) |
| 数据与 discovery | [AG-DCA](research/assistive-geometry-data-capability/README.md) · [AG-DUE](research/assistive-geometry-data-upgrade/README.md) · [候选事件挖掘](research/candidate-event-mining/README.md) |
| 历史路线/资料 | [USTRF-SC](research/ustrf-sc/README.md) · [Frontier Upgrade](research/frontier-upgrade-2026-07/README.md) · [研究资产](research/assets/README.md) · [组会总账](research/GROUP_MEETING_PROGRESS.md) |

前向研究默认采用 R4：`THESIS_DEVELOPMENT / PRODUCTION_PROMOTION` 两条权限分离；
只有显式启动 `PRODUCTION_PROMOTION` 才进入产品晋级门禁。

## 日期化快照与历史

以下文件用于复核当时结论，不是当前操作入口。

<details>
<summary>产品、平台和项目审计快照</summary>

- [HETEROGENEOUS_PLATFORM_P0A_R0_2026-07-31.md](HETEROGENEOUS_PLATFORM_P0A_R0_2026-07-31.md)
- [HETEROGENEOUS_PLATFORM_PHONE_P0A_R0_2026-07-31.md](HETEROGENEOUS_PLATFORM_PHONE_P0A_R0_2026-07-31.md)
- [PROJECT_AUDIT_2026-07-10.md](PROJECT_AUDIT_2026-07-10.md)
- [PROJECT_AUDIT_2026-07-28.md](PROJECT_AUDIT_2026-07-28.md)
- [PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md](PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md)
- [PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md](PUBLIC_VISUAL_INERTIAL_ROUTE_INTENT_2026-07-19.md)
- [EXPLICIT_ROUTE_INTENT_MODEL_CONTRACT_2026-07-19.md](EXPLICIT_ROUTE_INTENT_MODEL_CONTRACT_2026-07-19.md)
- [ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md](ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)
- [CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md](CORRIDOR_CAUSAL_PROGRESS_2026-07-20.md)
- [SECONDARY_MODEL_TEST_PLAN_CORRIDOR_CAUSAL_STUDENT_2026-07-16.md](SECONDARY_MODEL_TEST_PLAN_CORRIDOR_CAUSAL_STUDENT_2026-07-16.md)

</details>

<details>
<summary>SANPO 日期化研究快照</summary>

- [SANPO_GPU_UTILIZATION_2026-07-13.md](SANPO_GPU_UTILIZATION_2026-07-13.md)
- [SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md](SANPO_P0_SEED_FACTOR_AUDIT_2026-07-13.md)
- [SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md](SANPO_P1_LRASPP_ALIGNMENT_2026-07-13.md)
- [SANPO_P2_DETERMINISTIC_QUOTA_AUDIT_2026-07-13.md](SANPO_P2_DETERMINISTIC_QUOTA_AUDIT_2026-07-13.md)
- [SANPO_DETERMINISTIC_LINEAR_PROBE_2026-07-13.md](SANPO_DETERMINISTIC_LINEAR_PROBE_2026-07-13.md)
- [SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md](SANPO_P3_SPLIT_RECONSTRUCTION_2026-07-13.md)
- [SANPO_P3_OFFICIAL_TRAIN_FULL_DISCOVERY_2026-07-13.md](SANPO_P3_OFFICIAL_TRAIN_FULL_DISCOVERY_2026-07-13.md)
- [SANPO_P3_CONSENTED_CAPTURE_INTAKE_2026-07-13.md](SANPO_P3_CONSENTED_CAPTURE_INTAKE_2026-07-13.md)
- [SANPO_P3_VIEW_SOURCE_CONTRACT_2026-07-13.md](SANPO_P3_VIEW_SOURCE_CONTRACT_2026-07-13.md)
- [SANPO_EXTERNAL_EVENT_SOURCE_AUDIT_2026-07-15.md](SANPO_EXTERNAL_EVENT_SOURCE_AUDIT_2026-07-15.md)
- [SANPO_FEATURE_AND_DISTANCE_DIAGNOSTICS_2026-07-15.md](SANPO_FEATURE_AND_DISTANCE_DIAGNOSTICS_2026-07-15.md)
- [PUBLIC_SILVER_TRAINING_READINESS_2026-07-16.md](PUBLIC_SILVER_TRAINING_READINESS_2026-07-16.md)

</details>

开发日志月度历史：[2026-05](history/development-log/2026-05.md) ·
[2026-06](history/development-log/2026-06.md) · [2026-07](history/development-log/2026-07.md) ·
[2026-08](history/development-log/2026-08.md)。另见[早期项目材料](history/project-materials/README.md)
和[想法池历史](history/idea/README.md)。

## 维护规则

- 动态事实只更新 owning current，再从其他入口链接；导航页不复制状态和指标。
- 发布事实写入根 `CHANGELOG.md`；研究候选和失败实验写入研究结果或 archive。
- 新顶层文档、路线或聚合 archive README 必须同步索引并通过
  `scripts/check_docs_index.ps1`。
- 详细职责、历史保留和操作面预算以 [DOCUMENT_GOVERNANCE.md](DOCUMENT_GOVERNANCE.md) 为准。
