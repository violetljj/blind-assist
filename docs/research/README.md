# BlindAssist 研究入口

状态：`current / navigation-only`

本页只做稳定分类，不复制路线 token、指标、terminal 或 successor。先选择一个分类入口，
再进入一条路线 current；日期化 protocol/result、archive 和完整日志仅在复核历史时读取。

| 分类 | 回答的问题 | 唯一入口 |
|---|---|---|
| 因果诊断 | 当前失败首先发生在 target-policy stack 还是 representation？ | [D-ORACLE-1 current](failure-synthesis/README.md) |
| 算法 | 当前有哪些算法假设、各自状态与唯一下一动作？ | [算法研究入口](ALGORITHM_RESEARCH_CURRENT.md) |
| 数据 | 来源、truth、parent/session 独立性、coverage 和数据角色是否足以支持问题？ | [数据研究入口](DATA_RESEARCH_CURRENT.md) |
| 系统与平台 | 通信、延迟、导出、accelerator、设备和稳定性证据属于哪条工作流？ | [系统与平台入口](SYSTEM_RESEARCH_CURRENT.md) |

## 最小阅读规则

1. 一次只读一个分类 current 和一条明确路线/合同/测试入口；
2. 路线 README 是动态状态真源，分类入口只做摘要；
3. `snapshot`、`archive`、旧 handoff 和 consumed lock 不自动产生当前权限；
4. `UNKNOWN` 不是 negative，synthetic、teacher、source-derived、device 和用户证据不得互换；
5. 研究结果不自动修改默认 App、生产权限或安全结论。

上位边界见[研究治理](../RESEARCH_GOVERNANCE.md)，文档职责见[文档治理](../DOCUMENT_GOVERNANCE.md)。
