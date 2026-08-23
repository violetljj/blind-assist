# SUN3D referent-identifiability audit V0 result (2026-08-24)

状态：`READ_ONLY_AUTOPSY_COMPLETE / PUBLIC_GOAL_TO_PRIVATE_REFERENT_AMBIGUOUS / SELECTION_NOT_EVALUABLE / ACTIVE_REFERENT_SEARCH_NOT_AUTHORIZED_BY_THIS_EPISODE / NO_NEW_BENCHMARK_MODEL_CALL / NO_FRESH_DATA`

## 结论

封存 episode 的公开目标只有 `the door`，private evaluator 却只接受 `object 45`。只读审计证明两者没有公开可辨识
绑定：同一官方 sequence 的对象表还包含 `object 57 = door: bathroom`，而三个“object 45 可见且正确 proposal
存在”的帧都至少出现两扇视觉上合理的门。因此封存的 `selection accuracy | private object 45 visible + proposal =
0/3` 不能继续解释为 referent-selection algorithm failure。

若字面上的任意 door 都合法，合同应为 `SET_VALUED`，evaluator 不能只接受 object 45；若只允许 object 45，公开
goal 又没有提供区分它的关系、名称或 reference evidence，episode 应为 `AMBIGUOUS`。本审计按后一种 target-specific
评价语义签署：

`PUBLIC_GOAL_TO_PRIVATE_REFERENT_AMBIGUOUS_SELECTION_NOT_EVALUABLE`。

## 只读证据

审计重新读取同一官方 annotation，并严格匹配冻结 SHA-256
`9f6cf225411857420263b973bef968ac5d27e031a875d90f5314085ad78450e8`；未改 roster、pixels、proposal、Brain output
或 sealed report。`new benchmark model / provider / teacher / fresh episode = 0 / 0 / 0 / 0`；door 数量下界与
frame resolution 明确标为 `CODEX_VISUAL_REVIEWER_DERIVED_READ_ONLY_NOT_NATIVE_GT`，不冒充 source-native annotation。

| observation | private object 45 | 可见合理 door 下界 | frame resolution for object 45 | 正确 proposal rank |
|---|---|---:|---|---:|
| `sun3d-door-001` | visible | 2 | `AMBIGUOUS` | 2 |
| `sun3d-door-003` | visible | 1 | `UNIQUE` | none / proposal miss |
| `sun3d-door-013` | visible | 2 | `AMBIGUOUS` | 1 |
| `sun3d-door-015` | visible | 2 | `AMBIGUOUS` | 7 |

三个 private-target-absent confident commits（`002 / 007 / 008`）里也都至少有另一扇视觉上合理的门。因此
`wrong confident guidance | object 45 NOT_VISIBLE` 同样不能升级为 `wrong commit | public goal NOT_VISIBLE`。

审计程序验证 roster/final-report content hash、annotation hash、image hash、review coverage、native door-family objects、
correct proposal rank 与封存 outcome。正式本地 artifact：
`artifacts.local/evidence/sun3d-referent-identifiability-audit-v0/audit.json`；文件 SHA-256
`95ea8bfe5b790d8b1b74d15b1bd9fd9c07bb30dd1e0fa68bf4c43af495769493`，content SHA-256
`403bbe6a54bf97128d8382d8e2c5cb336b90db35df985369d8ed9448da117807`。

## Claim 修正与下一前门

- `object 45 VISIBLE=4 / NOT_VISIBLE=11` 只保留为 private-object descriptor，不代表公开 goal acquisition success/failure；
- `0/3 selection` 改为 `NOT_EVALUABLE_PUBLIC_REFERENT_NOT_IDENTIFIABLE`；
- `4/15 wrong confident guidance` 改为 `NOT_EVALUABLE_AS_PUBLIC_GOAL_WRONGNESS`；
- 本 episode 不授权 Active Referent Search、deterministic sweep、`H0=NOT_VISIBLE` policy、FSM、P1 或默认 App。

严格执行顺序在 Step 0 前门停止。若继续，必须先 outcome-blind 冻结一个 independently public-identifiable referent
contract，并在 pixels/provider output 前显式区分 `UNIQUE / SET_VALUED / AMBIGUOUS`，只对合同合法目标评分。不得用
private object ID、annotation label 或 evaluator truth 泄露给 provider，也不得回放或调参救本 episode。

Claim ceiling：
`CONSUMED_SUN3D_REFERENT_IDENTIFIABILITY_AUTOPSY_ONLY_NO_SELECTION_ACQUISITION_ACTIVE_SEARCH_CONTROL_SAFETY_OR_PRODUCT_CLAIM`。
