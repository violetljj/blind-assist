# AG-DCA R0 full-TRAIN capability atlas result

状态：`governed / COMPLETE / THREE_HYPOTHESES_NOT_SUPPORTED / AG_FCI_NOT_STARTED`

## 结论

全量 `16 parent × 300 = 4,800` TRAIN target capability atlas 已完成。当前 target 合同更适合
局部、稀疏、事件型几何监督；它不能支撑冻结的跨-parent right-censor、完整 2.5D grid，亦不能
提供 R2 决策所需的 complete factor truth 与 fresh paired authority。

| hypothesis | terminal | 关键原因 |
|---|---|---|
| `AG_QSF_H1_REOPEN` | `NOT_SUPPORTED_DATA` | right-censor 仅 `59` 帧、`1` parent、portrait-only；joint parent 仅 `1 < 4` |
| `AG_CBF_R0_STYLE_GRID` | `NOT_SUPPORTED_DATA` | full grid 仅 `196` 帧、`4` parent、`158/38` portrait/landscape，低于 `640 / 12 / 128 each` |
| `AG_FCI_R0_FOR_R2_DECISION` | `NOT_SUPPORTED_DATA_AND_AUTHORITY` | complete factor schema 和 truth-clear factor bundle 均为 `0`；joint parent 为 `0`；oracle injection 与 fresh paired outcome 未冻结 |

因此没有创建或启动 AG-FCI。该终态不否定因子反事实方法本身，只否定它在当前数据与权限快照下
作为 R2 选择证据的可执行性。

## Capability atlas

| capability | frames | parents | portrait | landscape |
|---|---:|---:|---:|---:|
| finite clearance event | 2,407 | 14 | 1,615 | 792 |
| right censor | 59 | 1 | 59 | 0 |
| ground plane valid | 3,424 | 16 | 2,144 | 1,280 |
| forward ground 0–2 m | 320 | 11 | 229 | 91 |
| forward ground 0–5 m | 216 | 5 | 178 | 38 |
| lateral observation ±0.5 m | 1,508 | 13 | 1,114 | 394 |
| lateral observation ±1 m | 1,099 | 11 | 907 | 192 |
| lateral observation ±2 m | 68 | 4 | 66 | 2 |
| full 2.5D grid | 196 | 4 | 158 | 38 |
| occupancy 1 m | 2,250 | 14 | 1,568 | 682 |
| occupancy 1.5 m | 2,210 | 14 | 1,543 | 667 |
| occupancy 2 m | 2,205 | 14 | 1,543 | 662 |
| consecutive source-order pair | 4,784 | 16 | 2,716 | 2,068 |
| explicit timestamp | 0 | 0 | 0 | 0 |
| pose transform | 0 | 0 | 0 | 0 |
| valid camera geometry | 3,424 | 16 | 2,144 | 1,280 |
| truth-clear | 59 | 1 | 59 | 0 |
| truth-occupied | 2,407 | 14 | 1,615 | 792 |

`consecutive source-order pair` 只证明同一 parent 内相邻 frame-stem 的顺序关系；由于显式 timestamp
和 pose transform 均未物化，它不能被提升为带物理时间或运动补偿的 temporal truth。

## FCI 数据与权限边界

单项 crisp factor 看似有较多支持：depth `4,767/16 parents`、ground `1,535/13`、support
`320/11`、obstacle `1,557/11`。但按冻结 joint contract 组合后：

- factor bundle 只有 `310` 帧，达到每 parent `32` 帧的仅 5 个 parent；
- truth-clear factor bundle 为 `0`，truth-occupied factor bundle 为 `300`；
- R2 depth uncertainty、support uncertainty、连续 obstacle-boundary truth 和 complete factor schema
  全部为 `0`；
- F0 deterministic reducer 已冻结并 PASS，但 real oracle-factor injection interface 与 fresh
  selection-eligible paired outcome 均不存在；
- B1 consumed Development 继续禁止重新包装成 R2 selection evidence。

因此 FCI 不是“尚未多跑几帧”，而是同时缺少数据对象与实验权限。

## 完整性收据

- 输入 manifest SHA-256：`A6F809C7...A7C2`；
- protocol SHA-256：`3E5DB541...F860`；requirements SHA-256：`4628C73A...A3C`；
- atlas：`23,817 bytes`，SHA-256 `12EB3B92...8DC7`；
- decisions：`6,881 bytes`，SHA-256 `3BA0445C...4BFD`；
- 全量执行 `41.002 s`；未读 RGB、模型、feature、checkpoint、模型 outcome、Development、
  Calibration 或 Confirmation，`UNKNOWN` 未被当作 negative。

## 终态与复用规则

AG-DCA R0 已完成且无活动 successor。checker 和 atlas schema 作为基础设施保留；未来 hypothesis
必须提交新的版本化 requirements，再对不可变 atlas 重放。不得修改本 R0 gate 来事后“救活”
QSF、CBF 或 FCI，也不得把 `SUPPORTED_FOR_PROTOCOL_LOCK`（若未来出现）解释成算法执行权限。
