# 数据研究入口

状态：`current / DEVELOPMENT_DATA_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|
| source identity、truth、coverage、quality、parent/session 独立性与 ancestry | `DEVELOPMENT_DATA_WORKSTREAM` | 本页；机器总账见 [JSON ledger](../../DATASET_MASTER_LEDGER.json)，缺口见 [DATASET_GAPS](../../DATASET_GAPS.md)，角色冲突见 [SOURCE_ROLE_CONFLICTS](../../SOURCE_ROLE_CONFLICTS.md) | `DATA_ROLE_AND_PARENT_DISJOINT_ADMISSION_SUCCESSOR`：先冻结数据合同、独立性和缺失策略，再为具体路线激活 admission 数据 |
| Assistive Geometry truth/source capability 与 hypothesis admission | `AG_DCA_R0_COMPLETE / QSF_CBF_NOT_SUPPORTED_DATA / FCI_NOT_SUPPORTED_DATA_AND_AUTHORITY` | [AG-DCA current](assistive-geometry-data-capability/README.md) | R0 无活动 successor；checker 与不可变 atlas 保留，新 hypothesis 必须提交版本化 requirements 后重放；AG-FCI 未启动 |
| Assistive Geometry gap-driven source upgrade | `AG-DUE_R1_SANPO_SYNTHETIC_AUDIT_PROTOCOL_LOCKED / EXECUTION_NOT_AUTHORIZED / FRAME_BODY_FORBIDDEN` | [AG-DUE current](assistive-geometry-data-upgrade/README.md) | `BLINDASSIST_ASSISTIVE_GEOMETRY_DUE_R1_SANPO_SYNTHETIC_METADATA_AND_OBJECT_INVENTORY_PREFLIGHT_EXECUTION`：只读 exact object inventory 与四个 metadata object；禁止 RGB/mask/depth body、Teacher 或训练 |
| 公开长视频候选事件发现 | `DISCOVERY_ONLY` | [候选事件挖掘](candidate-event-mining/README.md) | 只能产出 candidate pool；进入路线前另做 truth 与数据角色准入 |
| SANPO 数据与分割基线 | `INDEPENDENT_OR_COUPLED` | [SANPO current](../SANPO_CURRENT_STATUS.md) | 由 SANPO current 声明，不从本页推导算法晋级 |

## 边界

candidate、consumed、synthetic、teacher/pseudo-label 与 source GT 必须保持不同权威等级。
禁止用 candidate pool 反推 truth，或把被选择污染的数据重新包装为 confirmation holdout。
数据研究本身不改变默认 App。

总账体积很大，Codex 默认不得全文读取；先按 `source_id`、dataset、session 或 claim 用
`rg`/结构化查询定位，再读取命中附近和对应 contract。CSV 只用于表格交换，JSON 是机器查询入口。
