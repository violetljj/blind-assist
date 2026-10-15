# Documentation map / 文档导航

[Project overview](../README.md) · [中文项目介绍](../README.zh-CN.md) · [Code and directory map](CODE_MAP.md) · [Script index](../scripts/README.md) · [Tool index](../tools/README.md)

## Start with the task

| What you need / 要做什么 | Start here / 入口 |
| --- | --- |
| Build the app or make a first contribution | [English quickstart](QUICKSTART_EN.md) · [中文快速开始](QUICKSTART_ZH.md) · [Contributing](../CONTRIBUTING.md) |
| Run the current hardware or fusion demo | [硬件与融合演示](HARDWARE_OBSTACLE_DEMO.md) |
| Record Atom RGB, ToF and IMU; export a session | [手机采集与预览](PHONE_DATA_CAPTURE.md) |
| Find current priorities, capabilities and gaps | [项目一页状态](PROJECT_STATE.md) → [研究决定](CURRENT_DECISION.md) |
| Change Android code or locate its owner | [Code map](CODE_MAP.md) · [Isolated apps](../apps/README.md) |
| Verify a device change or prepare an APK | [Device regression](DEVICE_REGRESSION.md) · [Release and verification](RELEASE_AND_VERIFICATION.md) · [APK archive](APK_ARCHIVE.md) |

## Research routes

Read the matching `CURRENT.md` first; its links lead to detailed results,
protocols and reproduction records. The directory name `active/` alone does
not mean a route is currently advancing. Priorities belong to the
[research decision](CURRENT_DECISION.md), not this index.

| Area | Current status | Detailed records |
| --- | --- | --- |
| Hardware, calibration and mobile fusion | [Hardware current](../research/active/hardware-bringup/CURRENT.md) | [Hardware ledger](../research/active/hardware-bringup/README.md) · [Runs](../research/active/hardware-bringup/RUNS.md) |
| Cane-complementary forward perception / retained DTR history | [Forward perception current](../research/active/dtr-r0/CURRENT.md) | [DTR ledger](../research/active/dtr-r0/README.md) |
| Paused L10-R0 Goal-Lock evidence | [L10 current](../research/active/l10-r0/CURRENT.md) | [L10 ledger](../research/active/l10-r0/README.md) |

[Research workflow](../research/WORKFLOW.md) describes experiment execution;
the [knowledge reserve](../research/knowledge/README.md) indexes papers,
mechanisms, route-specific use and falsified hypotheses.

## Tools, hardware and local data

| Area | Guides |
| --- | --- |
| Hardware connection and firmware | [Glasses hardware route](GLASSES_HARDWARE_ROUTE.md) |
| Local paths, models, datasets and outputs | [Local artifacts](LOCAL_ARTIFACTS.md) · [Asset management](asset-management/README.md) · [Resource fabric and reuse](RESOURCE_FABRIC.md) |
| Configuration and machine setup | [Configuration map](../config/README.md) · [新电脑交接](NEW_COMPUTER_HANDOFF.md) |
| Compute and simulation | [Host research compute](HOST_RESEARCH_COMPUTE.md) · [CARLA](CARLA_PLAYBOOK.md) · [SkyDiscover](SKYDISCOVER_PLAYBOOK.md) |
| Recovery and reproducible checks | [关键证据备份与恢复](operations/CRITICAL_EVIDENCE_BACKUP.md) · [Python CI 与冻结文件检查](operations/PYTHON_CI.md) |

Command families are indexed in [tools](../tools/README.md) and
[shared research helpers](../scripts/research/README.md).

## Project material and history

- [Changelog](../CHANGELOG.md): version-specific changes and verification.
- [Model card](MODEL_CARD.md), [evaluation set](BLINDASSIST_EVALSET.md) and
  [detector benchmark](DETECTOR_BENCHMARK.md): model, data and measurement scope.
- [Threat model](THREAT_MODEL.md) and [security reporting](../SECURITY.md).
- [Demo guide](DEMO_GUIDE.md): prepare a presentation and distinguish the
  displayed app behavior from research results.
- [Documentation governance](DOCUMENT_GOVERNANCE.md),
  [research governance for formal claims](formal/RESEARCH_GOVERNANCE.md) and
  [operational runbooks](operations/CODEX_WORKFLOW.md).
- [History index](history-index.md): archived research and exact historical paths.

Closed research prose and historical snapshots retain their original evidence
and decisions; they do not replace the current route pages.
