# 数据研究入口

状态：`current / DEVELOPMENT_DATA_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|
| source identity、truth、coverage、quality、parent/session 独立性与 ancestry | `DEVELOPMENT_DATA_WORKSTREAM` | 本页；具体数据角色以对应 ledger/contract 为准 | `DATA_ROLE_AND_PARENT_DISJOINT_ADMISSION_SUCCESSOR`：先冻结数据合同、独立性和缺失策略，再为具体路线激活 admission 数据 |
| 公开长视频候选事件发现 | `DISCOVERY_ONLY` | [候选事件挖掘](candidate-event-mining/README.md) | 只能产出 candidate pool；进入路线前另做 truth 与数据角色准入 |
| SANPO 数据与分割基线 | `INDEPENDENT_OR_COUPLED` | [SANPO current](../SANPO_CURRENT_STATUS.md) | 由 SANPO current 声明，不从本页推导算法晋级 |

## 边界

candidate、consumed、synthetic、teacher/pseudo-label 与 source GT 必须保持不同权威等级。
禁止用 candidate pool 反推 truth，或把被选择污染的数据重新包装为 confirmation holdout。
数据研究本身不改变默认 App。
