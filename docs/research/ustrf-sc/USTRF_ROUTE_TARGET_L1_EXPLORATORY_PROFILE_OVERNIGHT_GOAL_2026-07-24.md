# USTRF route-target R2-L1E 单次探索 profile 通宵目标（2026-07-24）

## 目标

在不新增数据、不改变候选、不选择胜者的前提下，建立一条可恢复、哈希绑定、fail-closed 的独立运行链：严格读取 R2-L1 eligibility mask 的 62,229 帧显式 ledger，让冻结的 C1、C2、C3 各按完整 sequence 运行一次，只为当前已获授权的指标生成探索 profile，并用机器收据说明哪些结果有效、条件资格是否兑现、哪些仍不可评。

本阶段编号为 `R2-L1E`。它是 R2-L1 指标资格物化后的独立垂直切片，不修改父 V2 快照，不继承旧 selection runner 的选择权限，也不打开 Android shadow、H2、人体结论或生产授权。

## 冻结父输入

阶段开始时必须重新复算并精确绑定以下 SHA-256；任一不一致立即 `FAIL_CLOSED_INPUT_BLOCKED`：

| 输入 | SHA-256 |
| --- | --- |
| R2-L1 protocol | `4ab0c5dd687f7c9a3b791795271e4ecf5f23c1bbea41d2561503e6ce72e196ac` |
| eligibility mask | `b7dd5cfacc6f14153900bfaf811f3e76a1e188d1064013823f717f615e528157` |
| denominator receipt | `3f356ca69eb50bd176210d01bd9deb69e35acca46764b14603d0bb155d7b82bd` |
| validation receipt | `6bae841457ed0bd98cea6653e3c291b8cac5ba80603ec155811fcf95057196b9` |
| frozen C1–C3 implementation `candidates.py` | `82fb1a6391a6cb5fd5dd5116f16f26e250bb2421ee7211f57d691738176816c4` |

候选名册固定为：

1. `C1_CAUSAL_ROUTE_RELATION_FSM`
2. `C2_ROUTE_OCCUPANCY_EPISODE_FSM`
3. `C3_DUAL_KEY_CLEARANCE_FSM`

父 R2-L1 数量合同同时固定为：6,369 个 event/proposal unit、50,952 个 event×metric 分类、62,229 帧、41 条 sequence、62,188 个相邻 pair、3,801 个合格负暴露 pair、836 个合并区间及 `297,376,110,945ns` 严格负暴露。新阶段不得因本机可用文件不同而缩小这些集合。

## 开工前先冻结的新合同

先新增版本化配置 `configs/ustrf_route_target_l1_exploratory_profile_r1.json`，至少冻结：

- 上述五个父输入 SHA、候选顺序、候选实现 SHA；
- mask 中 62,229 行的精确 membership、sequence 顺序和 frame 顺序；
- detector raw stream、T0 association、causal route、capture timestamp 与 candidate consume timestamp 的字段合同；
- 每条 sequence 的状态初始化和唯一允许的 reset 边界；
- first-valid-complete-run、失败尝试、断点恢复及原子写入语义；
- 三种终态、指标权限、最小条件分母和禁止字段；
- 运行资源、超时、重试、检查点和 guard 收据要求。

旧 `run_crowdbot_holdout_candidates.py` 及 selection/scoring runner 不得直接复用为本阶段 runner：它们只处理旧 accepted event/window、混合 repeat/regeneration 语义、使用不同 evidence-age 口径且计算 winner。本阶段必须新增独立、receipt-aware 的探索 runner 和 validator，避免把旧选择权限带入。

## 分阶段执行

### A. 输入与实现 preflight

1. 重跑 R2-L1 validator，确认父收据仍为 `VALID`。
2. 验证 62,229 帧和 41 条 sequence 精确覆盖，无重复、缺口或顺序漂移。
3. 查明每帧是否已有 Android Canvas canonical detector raw tensor、冻结 T0 association 和候选所需 causal input。
4. 禁止用 host PIL reconstruction、未来 pose、annotation truth、候选输出或评分标签补候选输入。
5. 若 canonical stream 不完整，只允许按已经冻结的正式 App exporter 合同物化缺失流；不能改变模型、labels、`.35/.45`、preprocess、NMS、tracker 或候选。
6. 如果无法为全部 ledger 帧形成同口径输入，写出逐 source/sequence/frame/字段的 gap receipt 并终止，不运行任何候选。

### B. runner 与变异测试

实现由稳定 root adapter 调用，建议命令：

- `python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_metric_eligibility_exploratory_profiles_r2_l1.py`
- `python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_exploratory_profiles_r2_l1.py`
- `python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_exploratory_profiles_r2_l1.py`

在第一次候选运行前，测试至少证明：

- truth、clear、metric eligibility 和评分标签不会进入 candidate state transition；
- frame 缺失、乱序、跨 sequence 状态泄漏、重复运行、候选实现 SHA 漂移均 fail closed；
- 不允许 0/0 产生 pass/fail；
- 不允许 pre-clear 单元进入 clearance；
- 不允许 repeat truth pool 代替首次交付后的实际 repeat 分母；
- evidence age 少一帧即整项候选 profile 不可评，不能悄悄缩分母；
- L0 指标不能进入门禁、排名、winner 或推荐字段；
- profile/schema 中不存在 winner、rank、best candidate、tie-break 或 promotion。

### C. C1–C3 单次完整运行

- 按候选名册顺序执行，但三个候选相互隔离；每个候选对每条完整 sequence 只产生一条权威 trace。
- 候选状态只允许在 sequence 边界初始化；不得按 event/window 重置。
- truth join 必须发生在候选状态更新和输出落盘之后，仅用于归因与评分。
- 每个 candidate×sequence 写原子 checkpoint、输入 SHA、输出 trace SHA、首尾 frame、frame count、wall time、资源摘要与 attempt ID。
- 进程在 sequence 中途崩溃时，该 attempt 标为 incomplete 且没有评估权限；只可从该 sequence 起点恢复。第一个通过完整性验证的 trace 成为唯一权威 trace，禁止为改善结果选择性重跑。
- 不得以候选特定窗口、可评分事件子集或可用输入子集进行 replay。

### D. 分指标 profile 与总收据

对每个候选分别生成，不汇总排名：

| 指标 | 本阶段权限与口径 |
| --- | --- |
| `critical_miss` | L1；固定 8 个 critical event，报告点估计、分子/分母和来源族拆分 |
| `clearance` | L1；固定 12 个 terminal-clear event，严格排除 6,357 个 pre-clear 单元 |
| `unknown_or_stale_alert` | L1；固定全部 62,229 帧，报告 unknown/stale 状态上的 alert outcome |
| `repeat` | 条件 L1；实际分母只在首次交付且 episode 后续完整可观察后形成；若实际分母小于 5，标为 `evaluable_underpowered` 或 `not_evaluable`，不准通过 |
| `evidence_age` | 条件 L1；只有全部 62,229 帧同时存在 capture 与 consume timestamp 才有效，任一缺失使该候选整项 `not_evaluable` |
| `event_recall` | L0；因 alertable deadline 分母为 0，只允许记录诊断 raw count，不得形成 recall 成绩 |
| `regeneration` | L0；完整 post-clear 分母为 0，只允许诊断，不得 pass/fail |
| `false_alerts_per_minute` | L0；严格负暴露仅 4.956268516min，低于 5min floor；可输出冻结口径诊断计数，但不能获得 L1、门禁或候选比较权限 |

每项必须同时写分子、分母、资格状态、来源/来源族贡献、排除/删失原因、所用 mask row IDs/hash 和 claim boundary。任何条件资格未兑现都应如实下降为 `not_evaluable` 或 `evaluable_underpowered`，不能改阈值、改口径或补数据。

## 通宵运行的恢复与资源边界

- 所有大输出写入忽略的 `artifacts.local/evidence/ustrf-route-target-l1-exploratory-profile-r1/`，先写临时文件，完成 hash/fsync/结构校验后原子更名。
- 允许 sequence 边界断点恢复；恢复前逐个核验已有 checkpoint 和 trace SHA，不信任仅凭文件存在的完成标记。
- 对设备、GPU、外部进程采用有界重试；超限后写失败收据并结束，不无限等待、不静默重启。
- 默认使用 CPU。若冻结 canonical pipeline 必须使用设备或 GPU，复用仓库已经声明的 guard 与温度/内存/电源阈值；若尚无绑定阈值，必须在第一次重负载前先版本化冻结，不能运行后补写。
- 记录 wall time、最大 RSS、设备/GPU telemetry、退出码、系统事件和最后成功 checkpoint。guard 触发后禁止自动绕过。
- 不联网搜索、不下载新数据、不启动训练，也不运行与本阶段无关的实验。

## 唯一合法终态

### `EXPLORATORY_PROFILES_COMPLETE`

三个候选均有覆盖全部 41 条 sequence、62,229 帧的 first-valid-complete trace；所有 profile、收据和 validator 通过，且没有选择、排名或晋级结论。

### `FAIL_CLOSED_INPUT_BLOCKED`

父哈希、canonical input 或 frame membership 任一不完整。输出精确 gap matrix、已验证范围和阻塞首因；候选输出保持未授权。

### `FAIL_CLOSED_EXECUTION_ABORTED`

运行发生资源 guard、实现完整性、恢复一致性或合同错误。保留失败 attempt 和最后安全 checkpoint 的收据；任何部分 trace 均无 profile 权限。

不得使用“基本完成”“部分结果可参考”或从三个候选中只保留成功者作为第四种终态。

## 完成条件

只有同时满足以下条件，才可宣布 `EXPLORATORY_PROFILES_COMPLETE`：

1. 配置、runner、validator、测试与输出 schema 均有 SHA 绑定；
2. 父收据和 candidate implementation SHA 复核一致；
3. 三个候选各自覆盖 41/41 sequence 与 62,229/62,229 frame；
4. 每条权威 trace 可由 checkpoint、输入和输出 SHA 复算；
5. L1、条件 L1、L0 指标严格分栏，0/0、低样本和缺 timestamp 均 fail closed；
6. 输出不存在 winner、rank、best、tie-break、promotion 或 Android/H2/production authority；
7. validator 从冻结输入独立重建分母与 profile，并通过变异测试；
8. 形成日期化结果文档，明确数据资格、运行完整性、观察结果和不能声称的内容；
9. 独立 agent 做只读合同/实现/收据复核，所有阻断项已关闭；
10. 更新研究索引、模块 README 与 handoff；提交时只纳入本阶段文件，运行证据继续留在 ignored evidence 目录。

## 后继分流

本阶段结束后也不选择胜者：

- 若三者都完成：下一独立任务只审计 profile 是否暴露共同机制缺口，并决定是否值得为某一指标进入 L2 预注册；不得直接比较总分。
- 若条件 L1 未兑现：按实际缺的是首次交付、完整 episode 还是 consume timestamp，修执行/审计合同，不扩大下载范围。
- 若输入阻塞主要是真值、人物身份、路线或 canonical stream：优先修复标注和审计机制。
- 若只有 L0 指标缺分母：保持 L0，并为该指标单独提出最小数据补充边界；不得借其他指标的运行结果提升。

## 可直接启动的 `/goal`

```text
/goal 接下来完成 R2-L1E receipt-aware C1–C3 single-run exploratory profile closure。

严格绑定：
- R2-L1 protocol SHA-256 4ab0c5dd687f7c9a3b791795271e4ecf5f23c1bbea41d2561503e6ce72e196ac
- eligibility mask SHA-256 b7dd5cfacc6f14153900bfaf811f3e76a1e188d1064013823f717f615e528157
- denominator receipt SHA-256 3f356ca69eb50bd176210d01bd9deb69e35acca46764b14603d0bb155d7b82bd
- validation receipt SHA-256 6bae841457ed0bd98cea6653e3c291b8cac5ba80603ec155811fcf95057196b9
- candidates.py SHA-256 82fb1a6391a6cb5fd5dd5116f16f26e250bb2421ee7211f57d691738176816c4

先冻结独立探索配置、输入合同、恢复合同和 validator。对 mask 的 62,229 帧、41 条完整 sequence 做 fail-closed preflight；必须使用 Android Canvas canonical detector raw stream、冻结 T0 association 和 causal input，禁止 host PIL reconstruction、未来 truth 或候选输出进入状态更新。若全量输入不齐，生成精确 gap receipt 并以 FAIL_CLOSED_INPUT_BLOCKED 结束，不运行候选。

输入通过后，让 C1、C2、C3 各按完整 sequence 运行一次；状态只在 sequence 边界重置，truth 只在输出后归因。每个 candidate×sequence 原子写 checkpoint 和 trace SHA。中途失败的 attempt 无评估权限，只可从 sequence 起点恢复；第一个完整有效 trace 唯一权威，禁止候选特定重跑或 best-of-retries。

只生成分指标探索 profile，不选胜者、不排名、不晋级：
- L1：critical miss 固定 n=8；clearance 固定 n=12；unknown/stale 固定 62,229 帧。
- 条件 L1：repeat 只能使用首次交付后实际完整 episode 分母且至少 5；evidence age 必须覆盖全部 62,229 帧，缺一帧即整项不可评。
- L0：event recall、regeneration、false alerts/min 只能输出诊断，不得形成 pass/fail、门禁或比较。不得用 4.956268516min 负暴露冒充 5min L1 分母。

所有输出写入 artifacts.local/，哈希绑定、可恢复、可复算。采用有界重试和仓库既有资源 guard；输入、资源或完整性失败必须写机器收据并安全终止，不无限等待。完成前运行变异测试、完整 validator 和独立 agent 只读复核，并更新日期化结果、研究索引、模块 README 与 handoff。

唯一合法终态：
1. EXPLORATORY_PROFILES_COMPLETE；
2. FAIL_CLOSED_INPUT_BLOCKED；
3. FAIL_CLOSED_EXECUTION_ABORTED。

无论结果如何，保持 selection、Android shadow、H2、人体结果和生产权限关闭；不新增数据、不改模型/阈值/NMS/tracker/candidate、不运行训练。
```
