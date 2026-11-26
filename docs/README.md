# Documentation map / 文档导航

[Project overview](../README.md) · [中文项目介绍](../README.zh-CN.md) · [Code and directory map](CODE_MAP.md) · [Script index](../scripts/README.md) · [Tool index](../tools/README.md)

## 1. Core hot docs（日常高频入口）

| What you need / 要做什么 | Start here / 入口 |
| --- | --- |
| Find obstacle-line priorities, capabilities and gaps | [项目一页状态](PROJECT_STATE.md) → [研究决定](CURRENT_DECISION.md) |
| Run the current hardware or fusion demo | [硬件与融合演示](HARDWARE_OBSTACLE_DEMO.md) |
| Record Atom RGB, ToF and IMU; export a session | [手机采集与预览](PHONE_DATA_CAPTURE.md) |
| Change Android code or locate its owner | [Code map](CODE_MAP.md) · [Isolated apps](../apps/README.md) |
| Build the app or make a first contribution | [English quickstart](QUICKSTART_EN.md) · [中文快速开始](QUICKSTART_ZH.md) · [Contributing](../CONTRIBUTING.md) |
| Locate the separate thesis workspace and shared resources | [毕设迁移与共享](WORK_LINES.md) |

## 2. Research routes（按活跃与留证区分）

Read the matching `CURRENT.md` first; its links lead to detailed results,
protocols and reproduction records. The directory name `active/` alone does
not mean a route is currently advancing. Priorities belong to the
[research decision](CURRENT_DECISION.md), not this index. Thesis work lives in
its own workspace; see [work lines](WORK_LINES.md).

| Status | Line | Area | Current status | Detailed records |
| --- | --- | --- | --- | --- |
| Active | 避障 | EgoGeom-Omni wearable geometry foundation (with/without ToF) | [EgoGeom-Omni current](../research/active/egogeom-omni-v0/CURRENT.md) | [Route index](../research/active/egogeom-omni-v0/README.md) · [Wearable architecture](../research/active/egogeom-omni-v0/cane_blind_band_probe/CAMERA_IMU_OPT_TOF_ARCHITECTURE.md) |
| Active | 避障 | EgoClear pure-vision collision perception (v0–v14) | [EgoClear current](../research/active/egoclear-v0/CURRENT.md) | [Route index](../research/active/egoclear-v0/README.md) · [126-scene blind probe](../research/active/egoclear-v0/PHASE20_FRESH_UNSEEN_BLIND_PROBE_20261122.md) |
| Active | 避障 | Hardware, calibration and mobile fusion | [Hardware current](../research/active/hardware-bringup/CURRENT.md) | [Hardware ledger](../research/active/hardware-bringup/README.md) · [Runs](../research/active/hardware-bringup/RUNS.md) |
| Retained | 避障 | Cane-complementary forward perception / retained DTR history | [Forward perception current](../research/active/dtr-r0/CURRENT.md) | [DTR ledger](../research/active/dtr-r0/README.md) |
| Closed | — | Closed alert-policy & multi-frame memory exploration (`walk-alert-r0`) | [Stage summary (closed)](../research/active/walk-alert-r0/STAGE_SUMMARY_20261117.md) | [Walk-alert current](../research/active/walk-alert-r0/CURRENT.md) |
| Paused | — | Paused L10-R0 Goal-Lock evidence | [L10 current](../research/active/l10-r0/CURRENT.md) | [L10 ledger](../research/active/l10-r0/README.md) |

[Research workflow](../research/WORKFLOW.md) describes experiment execution;
the [knowledge reserve](../research/knowledge/README.md) indexes papers,
mechanisms, route-specific use and falsified hypotheses.

## 3. Hardware, compute, storage and device verification

| Area | Guides |
| --- | --- |
| Cloud GPU compute (`colabm` + `rclone`) & local worker | [Cloud compute runbook](operations/CLOUD_COMPUTE_COLABM.md) · [Host research compute](HOST_RESEARCH_COMPUTE.md) |
| Hardware connection and firmware | [Glasses hardware route](GLASSES_HARDWARE_ROUTE.md) |
| Device regression, release and APK archive | [Device regression](DEVICE_REGRESSION.md) · [Release and verification](RELEASE_AND_VERIFICATION.md) · [APK archive](APK_ARCHIVE.md) |
| Local paths, models, datasets and outputs | [Local artifacts](LOCAL_ARTIFACTS.md) · [Asset management](asset-management/README.md) · [Resource fabric and reuse](RESOURCE_FABRIC.md) |
| Configuration and machine setup | [Configuration map](../config/README.md) · [新电脑交接](NEW_COMPUTER_HANDOFF.md) |
| Simulation & search playbooks (specialized) | [CARLA](CARLA_PLAYBOOK.md) · [SkyDiscover](SKYDISCOVER_PLAYBOOK.md) |
| Recovery and CI checks | [关键证据备份与恢复](operations/CRITICAL_EVIDENCE_BACKUP.md) · [Python CI 与冻结文件检查](operations/PYTHON_CI.md) |

Command families are indexed in [tools](../tools/README.md) and
[shared research helpers](../scripts/research/README.md).

## 4. Governance, receipts, benchmarks and history

- [Changelog](../CHANGELOG.md): version-specific changes and verification.
- [Model card](MODEL_CARD.md), [evaluation set](BLINDASSIST_EVALSET.md) and
  [detector benchmark](DETECTOR_BENCHMARK.md): model, data and measurement scope.
- [YOLO export receipt](YOLO_EXPORT_PROVENANCE_RECEIPT.md) and
  [Android admission receipt](ANDROID_RESEARCH_PREVIEW_ADMISSION.md): model provenance and preview contracts.
- [Threat model](THREAT_MODEL.md), [security reporting](../SECURITY.md) and
  [public value](OPEN_SOURCE_PUBLIC_VALUE.md).
- [Demo guide](DEMO_GUIDE.md): prepare a presentation and distinguish displayed app behavior from research results.
- [Documentation governance](DOCUMENT_GOVERNANCE.md),
  [research governance for formal claims](formal/RESEARCH_GOVERNANCE.md) and
  [operational runbooks](operations/CODEX_WORKFLOW.md).
- [History index](history-index.md): archived research and exact historical paths.

Closed research prose and historical snapshots retain their original evidence
and decisions; they do not replace the current route pages.
