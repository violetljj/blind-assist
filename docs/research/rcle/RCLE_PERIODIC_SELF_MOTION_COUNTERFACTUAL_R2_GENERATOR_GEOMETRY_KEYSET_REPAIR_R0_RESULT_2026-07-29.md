# RCLE periodic self-motion counterfactual R2 P1 keyset repair result

日期：2026-07-29（Asia/Hong_Kong）

三轴结论：

| 维度 | 状态 | 含义 |
| --- | --- | --- |
| 科学状态 | `GEOMETRY_PASS` | G01–G14 14/14 PASS，几何实现成立 |
| 协议状态 | `VALID` | 当前 lock、manifest、hash、receipt 一致 |
| 执行权限 | `P2_NOT_AUTHORIZED` | P2 需要单独立项，不代表 P1 科学失败 |

组合终态：`GENERATOR_GEOMETRY_PASS / EXECUTION_NOT_AUTHORIZED`

这只关闭 P1 generator geometry。P2 仍须另立任务授权；本结果不产生 P3、P4、
RGB、Android、实时、产品或安全权限。

## 修正边界

不可变 R2 正式回执已经消费并保留为
`INVALID / INTERVENTION_NOT_EVALUABLE / HOLD_P1`。它的 14 项几何计算均
PASS，但 validator 把不可变 R0 receipt 的真实 evidence key
`producer_receipt.json` 错写为 `generator_receipt.json`。

因此历史 R2 应分轴阅读为：科学状态
`OBSERVED_GEOMETRY_GATES_PASS / CLAIM_NOT_SIGNABLE`、协议状态
`INVALID_KEYSET`、执行权限 `HOLD_P1`。`INVALID` 只否定该回执的协议可签署性，
不抹去已经观察到的几何计算。

本次另立
`RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_R2_KEYSET_REPAIR_R0`，
只做两类修正：

1. 按不可变 R0 receipt 冻结精确七键集合，其中使用
   `producer_receipt.json`；错误别名或增删键均失败。
2. generator evidence directory 采用 create-new-or-fail，正式 receipt 采用
   exclusive-create-or-fail；先只读预检为 `VALID`，再唯一写入正式回执。

没有改 seed、场景、renderer、环境、相机内参、轨迹、manifest schema、G01–G14、
阈值、R3 或三-pair 规则。新 all-seed manifest 与 R2 的 88 条记录逐字节一致，
其中前 80 条 MAIN 仍与 R0 逐字节一致。

## 冻结身份

| 对象 | SHA-256 |
| --- | --- |
| keyset-repair amendment | `f470291f700df41b1a69f652177faafc8e601fdfdf38b023c6de98462791abe4` |
| implementation lock | `a7fa41c0406908baf05805904111ba43fdbd8dd93b8c4e496706f1990438adc9` |
| generator source | `c0fd7b7de0e40e9d9eef1995da722c60f571b972a4669289d2e76e0d90ff8f72` |
| independent validator source | `14295ba9f3e08db5d3c6040d52261db9fdff7f62e95a93b15c5fadbcaa307058` |
| all-seed geometry manifest | `3dcf37496997a1edb2e47871c0dfc5185fd207016a26a86e29514412484e7ac6` |
| producer receipt | `641b2a30c30b07ea8b1f5ea899277fa1d9d69918095a8d4e1655f3bf95680264` |
| independent validation receipt | `95646437fbe0ef0cf03844f94467303f5d90ca15c3e22fc1785157b037a8c079` |

不可变前序身份：

| 版本 | 正式 receipt SHA-256 | 终态 |
| --- | --- | --- |
| R0 | `72e0b8e042be9eb6208389eb8d83e9e9e4ad28e54ec82f7064b5387cc1abd279` | `INTERVENTION_NOT_EVALUABLE / HOLD_P1` |
| R1 | `af00df05c115036ea31bb3d05addbebfcebad73122d2b354f7e52170c2277e9a` | `INTERVENTION_NOT_EVALUABLE / HOLD_P1` |
| R2 | `75899978919b67be260bbba1161d69ea09b42384f1730ec866a243f6d0f41a32` | `INTERVENTION_NOT_EVALUABLE / HOLD_P1` |

## 独立验证

只读预检与随后唯一一次正式验证均返回：

- `status=VALID`
- `terminal=GENERATOR_GEOMETRY_PASS`
- `state=EXECUTION_NOT_AUTHORIZED`
- `gate_pass_count=14`
- `gate_required_count=14`
- `failed_gates=[]`
- `errors=[]`

关键复算量：

| Gate | 结果摘要 |
| --- | --- |
| G01–G02 | 88 scene finite/multi-depth；80 MAIN grid-depth identities 全部通过 |
| G03 | 四 block 各 10,000 samples；RMS `4.8160e-14 px`，p99 `1.2711e-13 px` |
| G04–G07 | static、inverse-depth、single-homography rejection、rotation invariance 全部通过 |
| G08 | 160 motion sequence identities；865,440 samples；1,053 analytic disocclusions；visibility mismatch `0` |
| G09–G10 | 四 block pose hash 一致；endpoint closure 通过 |
| G11–G12 | 80 个六-arm cluster；pairing 与 source-known geometry identity 全部通过 |
| G13 | 16/16 arms；每条 602/602 frames 可见；inverse-depth increase `0.25`；integrated log-radial `0.2231435513` |
| G14 | base replay mismatch `0`；8 条 GUARD 双构建 replay mismatch `0` |

测试与静态验证：

- keyset-repair 定向测试：20/20 PASS；
- periodic self-motion counterfactual R2 模块测试：76/76 PASS；
- 变异覆盖包含错误键别名、目标/轨迹、pairing、geometry hash、reference hash、
  replay、generator 目录覆盖与正式 receipt 覆盖；
- validator 不导入 generator 或 RCLE。

独立只读终审再次核验：R0/R1/R2 receipt 身份无漂移，新 lock 的 14 个
source hash 与 9 个 evidence hash 均零漂移，正式 receipt 与上述终态一致。
审查记录一个非阻断完整性项：当前 validator 未把历史 R0 implementation lock
纳入未来漂移自动检测；R0 lock 当前实际 SHA 与 R0 receipt 记录值一致，因此不改变
本次 P1 终态。该项只进入 backlog，不再为它创建 P1 修订；P2 启动时如有必要做
一次只读确认即可。

## 最小治理规则

后续按变更性质决定工作量，避免治理重新压过算法研究：

| 变更类型 | 处理 |
| --- | --- |
| seed、数据、场景、相机、轨迹、算法、gate 或阈值变化 | 新科学版本，只重算受影响科学门 |
| manifest/receipt 身份错误，但科学输入和输出字节未变 | 薄修订，保留科学结论，只重验受影响协议门 |
| 文案、未来漂移监控或其他非阻断加固 | 进入 backlog，不阻断阶段、不创建新版本 |

每个阶段只保留一个当前 spec、一个 lock、一个 manifest、一个 receipt 和一个结果
入口；历史文件仅用于追溯，不再作为日常研究操作面。P1 至此关闭，不继续增加
receipt 基础设施或防御性 gate。

## 证据路径

- implementation lock：
  `docs/research/rcle/RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_GENERATOR_GEOMETRY_IMPLEMENTATION_LOCK_R2_KEYSET_REPAIR_R0_2026-07-29.json`
- all-seed manifest：
  `artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p1_geometry_r2_keyset_repair_r0/all_seed_geometry_manifest.jsonl`
- independent receipt：
  `artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p1_geometry_r2_keyset_repair_r0/independent_geometry_validation_receipt.json`

## 权限与下一步

已确认没有读取或运行 RCLE output，没有校准 blur/low-texture，没有运行 8 条
P3 性能预检，没有运行 480 MAIN 或 16 guardrail，没有修改 R3、阈值或三-pair，
也没有进入 sequence16、CoTracker、Android 或实时集成。

P1 到此停止。只有另立 P2 并保持现有 implementation lock、all-seed manifest 与
正式 receipt 身份，才可以开始 response-blind quality strength calibration。
