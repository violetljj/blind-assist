# 研究总入口

状态：`current`

先读本页，再进入一个分类入口。日期化结果、完整协议和历史流水不在这里展开。

## 当前研究结构

| 分类 | 当前重点 | 入口 |
|---|---|---|
| 算法 | 当前主线是 BlindAssist Assistive Geometry R2 因子化几何；F0 reducer 已 PASS，F1-P schema/loss/Kill Gate 已冻结，但现有 continuous-boundary/complete-factor supervision 前门不满足，F1 execution 仍未授权；B1-A0、AG-QSF 与 AG-CBF R0 保持各自负终态；DepthART-S 是可替换 encoder/initialization、depth baseline 与部署使能线 | [算法研究入口](ALGORITHM_RESEARCH_CURRENT.md) |
| 数据 | 数据集、truth、parent/session 独立性、coverage、质量和数据角色治理；AG-DCA R0 已完成 4,800-frame atlas，QSF/CBF 为 `NOT_SUPPORTED_DATA`、FCI 为 `NOT_SUPPORTED_DATA_AND_AUTHORITY`；AG-DUE R0 已锁 metadata-only gap prescreen，但尚无 source manifest/payload/Teacher authority | [数据研究入口](DATA_RESEARCH_CURRENT.md) |
| 系统与平台 | 通信链路、端到端延迟、性能优化、模型导出、设备部署和稳定性 | [系统与平台研究入口](SYSTEM_RESEARCH_CURRENT.md) |

## 阅读规则

- 分类入口只维护当前问题、状态、唯一真源和下一动作。
- 具体路线 README 只维护本路线，不代替其他分类入口。
- `snapshot` 是日期化结论，`archive` 是历史材料；二者都不自动产生当前执行权限。
- 任何研究结果都不自动修改默认 App、生产权限或安全结论。
