# BlindAssist 文档治理

状态：current
最后核验：2026-08-13
适用范围：仓库内所有协作者、自动化代理与长期任务。

## 目标

让下一位协作者能在不依赖聊天上下文的情况下找到唯一的当前规则、验证历史和下一步，同时保留旧记录的可追溯性。

## 文档职责与唯一真源

| 需求 | 唯一入口 | 不应承担该职责的文件 |
| --- | --- | --- |
| 新窗口项目状态、任务分类和最小读取路径 | `docs/PROJECT_STATE.md` | 全量 `README.md`、`DEVELOPMENT_LOG.md`、archive/snapshot |
| 当前产品能力、版本、最短构建入口 | `README.md` | 研究路线动态状态、`idea.md`、日期化实验报告 |
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
| 项目研究分类和阅读入口 | `docs/research/README.md` | 具体路线 README、日期化结果 |
| 算法路线摘要、唯一 successor 与默认 App 权限 | `docs/research/ALGORITHM_RESEARCH_CURRENT.md` | 数据/平台入口、日期化结果 |
| 跨路线系统研究分类 | `docs/research/SYSTEM_RESEARCH_CURRENT.md` | 与算法路线重叠的动态状态和 successor |
| 研究 Module 实时数量 | `scripts/research/MODULE_INDEX.md` 的机器校验 `N-of-N` 标记 | `scripts/README.md`、`REGISTRY.md` |
| 任务断点与工作区现场 | `artifacts.local/work/codex-handoffs/` | Git 提交文档 |

发生冲突时，以可复现的代码/门禁报告为事实基础；再以对应 `current` 协议为规则，以当前状态文档为操作摘要。研究阶段和证据传播以 `RESEARCH_GOVERNANCE.md` 为上位规则；领域协议可以更严格，但必须说明阶段、依据和最小 failure scope。`AI_REVIEW_GOVERNANCE.md` 覆盖旧 current/snapshot/handoff 中任何人工采集、标注、复核、仲裁或验收步骤；日期化快照只说明当时结论，不具有当前执行 authority。

## 更新规则

- 只改一个当前真源，再从其他入口链接它；不要复制会变化的数字、门禁结论或下一步。
- `README.md` 仅在产品、版本、构建入口或用户可见状态变化时更新；其“当前状态”只链接研究总入口，不复制路线名、阶段、指标或 successor。
- `CHANGELOG.md` 仅加入已发布或用户可见变化。候选模型、数据收集和失败实验写入研究记录，不伪装成 release note。
- `DEVELOPMENT_LOG.md` 只追加 durable decision、架构或 interface 变化、研究结论、
  重要验证、材料失败和可复用操作教训；普通小修复、一次性测试和常规重构不写条目。
  保留历史原文，不为美化时间线改写旧结论。
- `idea.md` 只保留待决方向。实验结束后写一条简短决策并链接证据，而不是复制实验流水。
- 新的顶层 `docs/*.md` 必须在 `docs/README.md` 中列为 `current`、`snapshot` 或 `archive`。`scripts/check_docs_index.ps1` 校验所有非归档 current、路线 README、protocol，以及 `history/archive` 内承担导航职责的聚合 `README*.md`；普通历史正文可以保留旧路径原文。
- 日常路线 `docs/research/<route>/README.md` 只保留当前主张、当前结论、少量证据入口、唯一 successor、允许/禁止和 claim ceiling，硬预算为 180 行。完整过程原文进入同路线 archive，不能从 archive 恢复 authority。
- 稳定 Module README 只描述 Interface、输出、安全边界和停止条件；轮次状态与 successor 委托给 owning route current。分类 current 的单行路线摘要应保持可读，不重新承载完整协议。
- 非历史 `docs/**/*.json` 中键为 `path` 或 `*_path` 的仓库相对稳定路径必须存在；URI、绝对路径、`artifacts.local/` 与易失 `build/` 证据不在此门内。
- `docs/PROJECT_STATE.md` 是冷启动导航，不复制研究结论；任务开始时先读它，默认读取一个分类 current/根入口和一个明确的路线/合同/测试入口，直接依赖、验证或冲突需要时可扩展。
- 冷启动导航只允许稳定身份、路径和读取规则；状态、主张、指标、successor、禁止动作和默认 App 权限必须只在对应 current 真源维护。
- `ALGORITHM_RESEARCH_CURRENT.md` 的 current 路线摘要必须与其唯一真源 README 的顶部 current 状态行、“唯一 successor”段和默认 App 标记同步；历史段落中偶然出现同名 token 不算同步。`SYSTEM_RESEARCH_CURRENT.md` 遇到同一 DepthART/HFTF 执行面时只分类并显式委托，不建立第二份动态真源。
- 研究 Module 数量只在 `scripts/research/MODULE_INDEX.md` 维护，结构门会将其 `N-of-N` 与 Git 可见 Module 目录实时对比；其他导航页不写数字副本。

## 历史与归档

- 不删除已经用于解释决策、版本或验证的历史文档。把它们标记为 `snapshot` 或 `archive`，并从当前入口链接。
- `CHANGELOG.md` 只保留 release/user-visible 变化；既有研究历史移入
  `docs/history/project-materials/` 后留一个可访问指针，不删除原文。
- 当 `DEVELOPMENT_LOG.md` 的近期部分再次影响查找效率时，按月复制旧日期块到 `docs/history/development-log/`；根文件保留索引和最近 2–4 周。迁移必须保持原文、日期和可访问链接。
- `scripts/check_project_structure.ps1` 将根日志限制为 6000 行、1200000 bytes，且最老日期不超过 28 天。行数与字节预算为近期中文工程记录保留合理余量；任一超限仍须完成月度原文归档，不能无限扩张根日志。
- 已完成任务从本地 handoff 索引移除前，先将持久决策写入相应 current 文档、CHANGELOG 或开发日志。

## 新任务的最小文档动作

| 任务类型 | 必需动作 |
| --- | --- |
| 小范围代码/文档修改 | 只在形成 durable decision 或重要验证时更新相关 current 文档或 `DEVELOPMENT_LOG.md` |
| 发布、演示或用户可见变化 | README + CHANGELOG + 发布验证文档 |
| Discovery/Canary 研究 | 最小实验只需问题/假设、可信 baseline、一个有意义变化、可观察指标/决策和停止条件；可在 scoped 输出中记录，不默认新增文档、contract、lock 或 receipt |
| Development 研究 | 沿用上述五项；复用论文/代码/模型/公开数据时记录来源、许可和继承/新贡献边界。只有结果真正改变下一决策或形成 durable claim 时才写一个 owning result，不强制 snapshot + 开发日志双写 |
| Confirmation/Deployment 或不可逆研究决定 | 使用 `RESEARCH_PROTOCOL_TEMPLATE.md` 的 STRICT profile，生成机器 contract 并运行 `scripts/validate_research_protocol.py` |
| 多阶段或跨窗口任务 | 遵循 `AGENTS.md` 的本地 handoff 规则 |
| 新协议、门禁或不可逆决定 | current 协议；必要时新增 `docs/decisions/ADR-XXXX-*.md` 并链接到 `docs/README.md` |

同一结论不得同时新增多份职责重叠的 snapshot、amendment、receipt 总结和 handoff。
研究总入口只保留分类导航；分类入口只保留当前问题、状态、唯一真源和 successor；详细研究史进入领域 archive，不在 current README 重复维护。
启动新研究路线时，必须先登记到 `docs/research/README.md` 对应分类入口；算法路线登记到
`docs/research/ALGORITHM_RESEARCH_CURRENT.md`，写明主张、状态、唯一真源、唯一 successor、
禁止动作和默认 App 权限。原路线若不再承担当前执行职责，立即降级为 `closed`、`paused`
或 `diagnostic`，其详细 README、协议和结果移入同域 `archive/`；不得让旧路线继续以“下一步”
形式悬挂在 current 页面。
优先更新一个 current 入口，并让详细机器证据留在 artifact。非阻断文案、命名和未来
审计便利不单独立项，不得成为算法研究 blocker。

## 前向维护完成标准

以后采用“新增即治理”：形成新职责的任务，必须在同一任务、同一提交内完成对应索引，
不能先制造孤儿文件，再把整理留给下一窗口。

| 变化 | 同步动作 |
| --- | --- |
| 新想法但尚未启动 | 只写 `idea.md`；不创建 active route 或虚构 successor |
| 新研究路线 | 更新研究分类 current；写明主张、状态、唯一真源、唯一 successor、禁止动作、默认 App 影响 |
| 新研究 Module | 创建最小 README；登记 `scripts/research/MODULE_INDEX.md`；在 `module_families.json` 中唯一分类 |
| 新 HFTF 文件 | 在 `roles.json` 中落入具体职责；`support` 必须保持 0 |
| 新稳定代码职责 | 更新 `docs/CODE_MAP.md`；只写稳定入口，不复制运行状态 |
| 新顶层文档或稳定脚本 Interface | 更新 `docs/README.md` 或 `scripts/README.md` 对应索引 |
| 路线关闭、暂停、诊断化或 successor 改变 | 同提交更新 current 真源；详细过程归档，旧结果不得继续挂在“下一步” |
| 默认 App 影响改变 | 同提交更新产品 current 与路线 current；研究结果不得自行获得产品权限 |

任何需要依靠全仓搜索才能发现的新稳定职责，都视为治理未完成。动态信息只修改拥有它的
current 真源；导航页只增加或调整链接，不复制状态和指标。

## 验证

文档变更只运行覆盖实际改变面的最小检查。例如导航、current、protocol 或聚合 archive
README 变化执行：

```powershell
git diff --check
pwsh -File scripts/check_docs_index.ps1
```

只有 current 状态/successor、项目结构或 Module 合同变化时才追加
`check_project_structure.ps1`。修改门禁本身时运行其直接测试；涉及脚本入口、发布或工具链时，
按 owning 文档追加专项验证。普通提交或 push 不要求仓库卫生门，也不等待远端 CI。
