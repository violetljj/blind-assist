# BlindAssist 项目冷启动状态

状态：`current / NAVIGATION_ONLY`
最后核验：2026-08-08

> 新窗口首先读本页。目标是 30 秒内确定项目是什么、这次任务该读哪三个文件。
> 本页只做导航，不复制实验历史；动态结论以表中链接的 current 真源为准。

## 项目是什么

BlindAssist 是一个本地 Android 助盲避障原型：手机 CameraX 采集，视觉模型推理，
规则层生成语音、震动和 Compose 反馈。它不是安全认证设备，不能替代盲杖、导盲犬或人工判断。

## 动态真源

本页不复制会变化的路线、状态、指标、下一步或权限。只按任务类型打开一个 current
入口；该入口再链接唯一路线真源。

| 范围 | 动态真源 |
|---|---|
| 算法 | [算法 current](research/ALGORITHM_RESEARCH_CURRENT.md) |
| 数据 | [数据 current](research/DATA_RESEARCH_CURRENT.md) |
| 系统与平台 | [系统 current](research/SYSTEM_RESEARCH_CURRENT.md) |
| 产品、默认 App、构建 | [根 README](../README.md) |
| 未决想法 | [idea.md](../idea.md) |

如果本页与上述入口不一致，以对应 current 真源和可复现门禁为准；不要在本页修补动态结论。

## 这次任务只读哪三个文件

| 任务关键词 | 先读的三个文件 |
|---|---|
| 算法、模型、DepthART、DA2、双环 | 本页 → [算法 current](research/ALGORITHM_RESEARCH_CURRENT.md) → 对应路线 README |
| 数据集、truth、标注、split、coverage | 本页 → [数据 current](research/DATA_RESEARCH_CURRENT.md) → 对应 data ledger/contract |
| 延迟、链路、性能、QNN、HTP、部署 | 本页 → [系统 current](research/SYSTEM_RESEARCH_CURRENT.md) → 对应 benchmark/preflight |
| Android、App、CameraX、UI、构建 | 本页 → [根 README](../README.md) → 对应模块 README 或测试 |
| 文档、索引、归档、冷启动 | 本页 → [文档治理](DOCUMENT_GOVERNANCE.md) → [docs 索引](README.md) |
| 新想法、暂不推进 | 本页 → [idea.md](../idea.md) → 原始方案/调研文件 |

“对应”只允许打开分类入口中明确链接的一个 current、contract、ledger 或 benchmark；
不要为了寻找背景自动扫描整个 `docs/` 或 `scripts/`。

## 默认不要读

除非任务明确要求历史追溯、复现或审计，否则跳过：

- `docs/**/archive/`
- 日期化 `snapshot`、旧实验报告、完整研究史
- `DEVELOPMENT_LOG.md` 全文、`docs/history/`
- `artifacts.local/`、`build/`、`.cxx/` 和其他机器产物
- 与当前任务无关的研究域、数据台账和共享工作树改动

## 冷启动规则

1. 先读本页，给任务归类。
2. 只读对应分类 current 和一个明确的路线/合同/测试入口。
3. 先检查 `git status --short`，把已有改动视为用户所有。
4. 只有遇到链接缺失、状态冲突或用户要求历史时，才扩大读取范围。
5. 完成后把新结论写回唯一 current；未决定的想法留在 `idea.md`，不要创建伪 current。

## 同步契约

- 研究状态、主张、successor、禁止动作或默认 App 权限变化：同一提交更新对应 current 真源。
- 分类入口只维护本分类摘要；路线 README 只维护本路线摘要；本页不重复这些内容。
- 新增或移动 current 文档：同一提交更新本页（仅入口）和 `docs/README.md`，再运行索引检查。
- snapshot、archive、日志和外部方案不会自动改变 current；采用前必须显式登记并写明 successor。
