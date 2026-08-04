# FRESH-TF 及已开拓后继路线暂停记录

日期：2026-08-05

状态：`FRESH_TF_AND_OPENED_SUCCESSORS_PAUSED_BY_USER / PAUSED_NO_ACTIVE_EXECUTION`

## 用户决定

FRESH-TF 及本轮由它开拓出的后继路线暂时全部暂停，不再推进。暂停覆盖：

- whole-frame freshness 与 R1-A local geometric validity 的任何 successor；
- dense/fixed-cell depth propagation 的任何改版；
- fresh metric snapshot layered-intrusion R0 的 18-session 采集、admission 和 outcome；
- PMAF、HSTF-PMA、periodic metric anchoring、stable Track metric anchoring；
- 基于这些路线的风险触发 NPU 调度和 App 集成。

暂停期间不得继续采集新 cohort、运行 arm outcome、修改协议/阈值/roster、开发新
successor、为这些路线采购硬件，或将其晋级到研究主线、App、提醒、生产、导航和安全
路径。

## 保留边界

暂停不是失败终态，不覆盖或改写已有负结果，也不删除证据。以下内容继续完整保留：

- HFTF 作为用户和本项目已经建立的原创贡献；
- 所有冻结协议、结果、receipt、实现、测试和 hash；
- 已验证的 CameraX 同帧 QNN 米制深度与 NPU 工程结果；
- 现有 depth experience demo、默认 BlindAssist App 和无关研究路线。

fresh-snapshot R0 在任何正式来源采集和 QNN outcome 打开前暂停，因此其 18-session
protocol 仍是未消费设计，不产生支持、失败或 `NOT_EVALUABLE` 科学终态。

只有用户以后明确指定恢复哪一条路线和范围时才可重启。重启前必须先重新审计仓库、
source freshness、硬件状态和证据权限，不能从本暂停记录自动延续执行。
