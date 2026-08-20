# 数据研究入口

状态：`current / DEVELOPMENT_DATA_WORKSTREAM / PARALLEL_OR_COUPLED`

## 当前总表

| 当前问题 | 状态 | 唯一真源 | 唯一 successor |
|---|---|---|---|
| source identity、truth、coverage、quality、parent/session 独立性与 ancestry | `DEVELOPMENT_DATA_WORKSTREAM` | 本页；机器总账见 [JSON ledger](../../DATASET_MASTER_LEDGER.json)，缺口见 [DATASET_GAPS](../../DATASET_GAPS.md)，角色冲突见 [SOURCE_ROLE_CONFLICTS](../../SOURCE_ROLE_CONFLICTS.md) | `DATA_ROLE_AND_PARENT_DISJOINT_ADMISSION_SUCCESSOR`：先冻结数据合同、独立性和缺失策略，再为具体路线激活 admission 数据 |
| Assistive Geometry truth/source capability 与 hypothesis admission | `AG_DCA_R0_COMPLETE / QSF_CBF_NOT_SUPPORTED_DATA / FCI_NOT_SUPPORTED_DATA_AND_AUTHORITY` | [AG-DCA current](assistive-geometry-data-capability/README.md) | R0 无活动 successor；checker 与不可变 atlas 保留，新 hypothesis 必须提交版本化 requirements 后重放；AG-FCI 未启动 |
| Assistive Geometry gap-driven source upgrade | `AG-DUE_R1_SANPO_SYNTHETIC_PREFLIGHT_NOT_EVALUABLE / EXACT_METADATA_OBJECT_MISSING / ROUTE_CLOSED` | [AG-DUE current](assistive-geometry-data-upgrade/README.md) | `NONE_STOP_AT_PREFLIGHT_TERMINAL`：exact annotation-type object HEAD=404，禁止 fallback/path guessing；新 source/session/path 只能从另行版本化 R0 manifest 与 source-specific protocol 重新准入 |
| 公开长视频候选事件发现 | `DISCOVERY_ONLY` | [候选事件挖掘](candidate-event-mining/README.md) | 只能产出 candidate pool；进入路线前另做 truth 与数据角色准入 |
| BA-ADT Goal Episode mining | `REVERSIBLE_EXPLORATION / ADT_0_FULL_SEQUENCE_TARGET_SELECTED / SEQ136_CARROT_RGB_AND_GT_ACQUIRED / ADT_1_FLOW5_TRACKER_DEVELOPMENT_ADMITTED / ADT_2_DEVELOPMENT_DEMO_RENDERED` | [Goal Copilot current](goal-copilot/README.md)；[sample result](goal-copilot/BA_ADT_REAL_EVIDENCE_ADT0_SAMPLE_RESULT.md)；[selection/evaluation/demo result](goal-copilot/BA_ADT_REAL_EVIDENCE_ADT0_SELECTION_ADT1_CANARY_RESULT.md)；实现见 [`ba_adt_real_evidence`](../../scripts/research/ba_adt_real_evidence/README.md) | 保持 GT 与 RGB estimator 严格分离；下一步只处理同一 consumed Development sequence 的长 dropout 与 instance-conditioned redetection |
| SANPO 数据与分割基线 | `INDEPENDENT_OR_COUPLED` | [SANPO current](../SANPO_CURRENT_STATUS.md) | 由 SANPO current 声明，不从本页推导算法晋级 |

## 边界

candidate、consumed、synthetic、teacher/pseudo-label 与 source GT 必须保持不同权威等级。
禁止用 candidate pool 反推 truth，或把被选择污染的数据重新包装为 confirmation holdout。
数据研究本身不改变默认 App。

总账体积很大，Codex 默认不得全文读取；先按 `source_id`、dataset、session 或 claim 用
`rg`/结构化查询定位，再读取命中附近和对应 contract。CSV 只用于表格交换，JSON 是机器查询入口。
