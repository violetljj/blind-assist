# 当前研究决定

更新：2026-09-24

Status: `L10_R0_PAUSED / DTR_R2_DYNAMIC_RETAINED`

## 范围

论文主体与唯一推进的算法主线为盲杖互补的前视障碍感知。保留 A 基线和 A+LOCAL 的收益与成本；当前 Android 实验适配不等于实机准确率获证。读者先看[中文项目状态](PROJECT_STATE.md)，再看[避障当前页](../research/active/dtr-r0/CURRENT.md)。

[L10](../research/active/l10-r0/CURRENT.md) 正式暂停：保留代码、冻结结果和复现入口，不再自动启动实验。TARO、SANPO、PanoLab 和语义锚点只作历史成果或有明确用途的素材，不另开并行研究。是否作为论文扩展章节取决于写作需要；不承诺新增一章。

## 工作顺序

1. 已完成选定关键模型、基准与评估证据的副机备份及逐文件恢复校验；[范围与回执](operations/CRITICAL_EVIDENCE_BACKUP.md)明确未覆盖项，不能宣称全论文依赖已备份。
2. 已整理短中文状态入口并接入 Python CI；本地工具测试与 Windows 检出检查通过，Ubuntu 执行仍由 CI 覆盖。[测试与历史哈希差异](operations/PYTHON_CI.md)。
3. 提醒策略作为避障的有界工作包：固定检测输出，比较首次提醒延迟、重复提醒及关键事件被抑制次数；实际疲劳与 TalkBack 配合仍需交互验收。本次整理不启动新提醒实验。

## 决策和验证

新问题、额外预算、判定规则变化先交用户决定。用户已授权问题内，由执行者完成实现、必要机械修复、针对性验证和交付，不逐条请求许可。失败的冻结实验保留失败，不能事后换阈值/子集挽救；不同机制的新问题另行明确。

探索只保留问题、对照、判定、结果、下一步和必要身份信息。独立审计用于结论关键或存在具体完整性风险的工作，不是每次小实验的默认环节；自动断言总数不代表独立证据数量。正式论文数字和受保护数据继续遵守[正式治理](formal/RESEARCH_GOVERNANCE.md)。

`UNKNOWN`/`NOT_EVALUABLE` 不等于无障碍；Development 复用不恢复数据新鲜性。组件收益不自动成为整个系统收益。

## 已关闭的流程问题

ledger303 的原始回执已于 2026-09-22 恢复，并完成指定 operating-scope 实验登记。[修复记录](../research/active/dtr-r0/nearfield/LEDGER_REPAIR_20260922.md)只覆盖所述记录，不代表所有历史待登记项已补齐。历史快照内的 pending 是当时状态，不能作为今天的故障判断。

[整理前正文](operations/snapshots/CURRENT_DECISION_20260924.md)保存历史数字与决定；不作为新实验授权。
