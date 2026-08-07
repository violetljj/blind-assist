# BlindAssist 文档治理

状态：current
最后核验：2026-07-30
适用范围：仓库内所有协作者、自动化代理与长期任务。

## 目标

让下一位协作者能在不依赖聊天上下文的情况下找到唯一的当前规则、验证历史和下一步，同时保留旧记录的可追溯性。

## 文档职责与唯一真源

| 需求 | 唯一入口 | 不应承担该职责的文件 |
| --- | --- | --- |
| 当前产品能力、版本、最短构建入口 | `README.md` | `idea.md`、日期化实验报告 |
| 已发布或用户可见变化 | `CHANGELOG.md` | 研究实验日志 |
| 当前 SANPO 状态、硬门、禁止事项、下一步 | `docs/SANPO_CURRENT_STATUS.md` | `idea.md`、`CHANGELOG.md` |
| 当前双环论文系统路线、阶段−1门、权限与下一步 | `docs/research/dual-loop/README.md` | `idea.md`、日期化讨论或实验报告 |
| 暂停的 RCLE 科学/协议终态、保留权限与禁止事项 | `docs/research/rcle/README.md` | `AGENTS.md`、根 `README.md`、`docs/SANPO_CURRENT_STATUS.md`、`scripts/README.md` |
| 研究阶段、冻结强度、失败学习、规则质疑和证据复用 | `docs/RESEARCH_GOVERNANCE.md` | 单轮 prereg、日期化结果、旧 handoff |
| 端到端自主工作流与禁止人工前置条件 | `docs/AI_REVIEW_GOVERNANCE.md` | 日期化 snapshot、旧 handoff、历史实验合同 |
| 当前操作协议与安全门 | `docs/README.md` 标为 `current` 的对应文件 | 日期化 snapshot |
| 近期工程改动与验证 | `DEVELOPMENT_LOG.md` | README、CHANGELOG |
| 尚未决定的方向 | `idea.md` | 当前状态文档、开发日志 |
| 日期化实验、审计与研究结论 | `docs/*_YYYY-MM-DD.*` 或 `docs/research/` | current 协议 |
| 跨研究域算法路线摘要、唯一 successor 与默认 App 权限 | `docs/research/ALGORITHM_ROUTE_REGISTRY.md` | 各领域详细 README、日期化结果 |
| 任务断点与工作区现场 | `artifacts.local/work/codex-handoffs/` | Git 提交文档 |

发生冲突时，以可复现的代码/门禁报告为事实基础；再以对应 `current` 协议为规则，以当前状态文档为操作摘要。研究阶段和证据传播以 `RESEARCH_GOVERNANCE.md` 为上位规则；领域协议可以更严格，但必须说明阶段、依据和最小 failure scope。`AI_REVIEW_GOVERNANCE.md` 覆盖旧 current/snapshot/handoff 中任何人工采集、标注、复核、仲裁或验收步骤；日期化快照只说明当时结论，不具有当前执行 authority。

## 更新规则

- 只改一个当前真源，再从其他入口链接它；不要复制会变化的数字、门禁结论或下一步。
- `README.md` 仅在产品、版本、构建入口或用户可见状态变化时更新。
- `CHANGELOG.md` 仅加入已发布或用户可见变化。候选模型、数据收集和失败实验写入研究记录，不伪装成 release note。
- `DEVELOPMENT_LOG.md` 只追加有实际项目变化的简洁条目；保留历史原文，不为美化时间线改写旧结论。
- `idea.md` 只保留待决方向。实验结束后写一条简短决策并链接证据，而不是复制实验流水。
- 新的顶层 `docs/*.md` 必须在 `docs/README.md` 中列为 `current`、`snapshot` 或 `archive`；运行 `scripts/check_docs_index.ps1`。

## 历史与归档

- 不删除已经用于解释决策、版本或验证的历史文档。把它们标记为 `snapshot` 或 `archive`，并从当前入口链接。
- 当 `DEVELOPMENT_LOG.md` 的近期部分再次影响查找效率时，按月复制旧日期块到 `docs/history/development-log/`；根文件保留索引和最近 2–4 周。迁移必须保持原文、日期和可访问链接。
- `scripts/check_project_structure.ps1` 将根日志限制为 6000 行、1200000 bytes，且最老日期不超过 28 天。行数与字节预算为近期中文工程记录保留合理余量；任一超限仍须完成月度原文归档，不能无限扩张根日志。
- 已完成任务从本地 handoff 索引移除前，先将持久决策写入相应 current 文档、CHANGELOG 或开发日志。

## 新任务的最小文档动作

| 任务类型 | 必需动作 |
| --- | --- |
| 小范围代码/文档修改 | 相关 current 文档或 `DEVELOPMENT_LOG.md` 二选一，按职责更新 |
| 发布、演示或用户可见变化 | README + CHANGELOG + 发布验证文档 |
| Discovery/Canary 研究 | 优先在现有 current 或单个 LITE round 记录中写问题、来源、最小实验、结果和下一步；不默认新增 contract/lock/receipt |
| Development 研究 | 一个结果 snapshot + 简短开发日志；仅在身份、冻结或重放风险需要时增加机器 contract |
| Confirmation/Deployment 或不可逆研究决定 | 使用 `RESEARCH_PROTOCOL_TEMPLATE.md` 的 STRICT profile，生成机器 contract 并运行 `scripts/validate_research_protocol.py` |
| 多阶段或跨窗口任务 | 遵循 `AGENTS.md` 的本地 handoff 规则 |
| 新协议、门禁或不可逆决定 | current 协议；必要时新增 `docs/decisions/ADR-XXXX-*.md` 并链接到 `docs/README.md` |

同一结论不得同时新增多份职责重叠的 snapshot、amendment、receipt 总结和 handoff。
算法路线总表只保留短摘要和唯一 successor；详细研究史进入领域 archive，不在 current README 重复维护。
启动新算法路线时，必须先建立短 current 入口并登记到
`docs/research/ALGORITHM_ROUTE_REGISTRY.md`，写明主张、状态、唯一真源、唯一 successor、
禁止动作和默认 App 权限。原路线若不再承担当前执行职责，立即降级为 `closed`、`paused`
或 `diagnostic`，其详细 README、协议和结果移入同域 `archive/`；不得让旧路线继续以“下一步”
形式悬挂在 current 页面。
优先更新一个 current 入口，并让详细机器证据留在 artifact。非阻断文案、命名和未来
审计便利不单独立项，不得成为算法研究 blocker。

## 验证

文档变更至少执行：

```powershell
git diff --check
pwsh -File scripts/check_docs_index.ps1
pwsh -File scripts/check_project_structure.ps1
```

涉及脚本入口、发布或工具链时，按对应专项文档补充验证。
