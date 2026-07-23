# USTRF route-target R2-L1 指标资格物化结果（2026-07-23）

## 结论

本轮完成的是现有数据的“指标资产负债表”，不是算法成绩。候选盲物化覆盖 LILocBench 15 个事件、CrowdBot 首组 6,340 个已被取代的非视觉提案和 replacement 14 个事件，共 6,369 个单位；每个单位均有 8 个指标分类，合计 50,952 个分类单元。

- 已具备 `L1_EXPLORATORY_ELIGIBLE`：`critical_miss`、`clearance`、`unknown_or_stale_alert`。
- 仅具备 `L1_CONDITIONAL_ON_CANDIDATE_OBSERVATION`：`repeat`、`evidence_age`。
- 仍停在 `L0_ENGINEERING_DIAGNOSTIC`：`event_recall`、`regeneration`、`false_alerts_per_minute`。

因此下一步可另开独立任务，让 C1–C3 各运行一次并只生成探索 profile；本轮没有运行或读取 C1–C3、没有选择胜者、没有新增数据，也没有开放 selection、confirmation、Android shadow、H2、人体结果或生产权限。

父 V2 配置是不可变治理快照，仍保留 `highest_authorized_level=L0` 和 `r2_metric_eligibility_masks_frozen=false`，本轮没有原地改写它。这里的逐指标 L1 资格由新增、版本化且哈希绑定的 R2-L1 receipt 给出；下一任务必须显式绑定该 receipt，不能把父快照的状态字段手改为通过。

## 逐指标资产负债表

| 指标 | 当前有效分母/支持池 | 来源族贡献 | 资格 | 主要缺口 |
| --- | ---: | --- | --- | --- |
| event recall | `0` events | 无 | L0 | 6,369 个单位均没有冻结的 `alertable_deadline`；不得用 alertable start 或 clear 猜测 |
| critical miss | `8` critical events | CrowdBot `5/8`；LILocBench `3/8` | L1 | 只能评分已重建为唯一、连续至少 3 帧、同一人物且 causal route 全 known 的 critical interval |
| repeat | 实际候选分母 `null`；preoutput truth pool `12` | CrowdBot `2/12`；LILocBench `10/12` | 条件 L1 | 必须先发生候选首次交付；truth pool 不能冒充实际分母，实际分母仍须至少 5 |
| clearance | `12` events | CrowdBot `2/12`；LILocBench `10/12` | L1 | pre-clear、clear 未被同一人物身份覆盖或 1.5 秒 follow-up 不足均排除 |
| regeneration | `0` complete post-clear intervals | 无 | L0 | 没有同一人物身份在 clear 后连续可观察满 2 秒的区间 |
| false alerts/min | `297,376,110,945ns = 4.956268516min` | CrowdBot `65.8640%`；LILocBench `34.1360%` | L0 | 距冻结的 5 分钟 L1 floor 还差 `2.623889055s`；不得用 route unknown、人物真值不完整或 active truth 帧补分母 |
| evidence age | `62,229` preoutput frames | CrowdBot `92.6176%`；LILocBench `7.3824%` | 条件 L1 | 候选运行必须为全部 mask 帧提供 consuming timestamp；缺一帧即整项 profile 无效，不能缩分母 |
| unknown/stale alert | `62,229` preoutput frames | CrowdBot `92.6176%`；LILocBench `7.3824%` | L1 | route validity 已逐帧齐全；候选运行后仍须给全部 mask 帧绑定 alert outcome |

这里的 L1 只说明现有真值/时间/路线支持足以生成探索性点估计和区间，不说明候选表现好，也不允许横向选胜者。

## 事件、clear 与删失

- 事件宇宙：LILocBench `15`；CrowdBot 首组 superseded proposal `6,340`；CrowdBot replacement `14`。
- receipt 中有 `3,139` 个原始 clear marker，但只有 `12` 个满足同一人物 terminal-clear 观察合同，也只有这 `12` 个满足 clearance follow-up 资格。原始 clear marker 不能直接当作 terminal-clear 可评分分母。
- terminal-clear observability 的 recall-eligible 分母是 `0`，结果严格为 `0/0 -> not_evaluable`，不是 0、通过或失败。
- CrowdBot 首组 6,340 个提案全部保留在逐指标清单中，但因非视觉提案已被取代、相机绑定人物身份和 all-person route-role truth 不完整，不进入有效事件分母。
- 每个指标的排除原因与 censor state 均在 mask 逐事件记录，并在 denominator receipt 复算聚合。clearance 的 `6,357` 个 pre-clear 单元全部为 `not_evaluable_pre_clear`，不能混入 survival censor；repeat 的 17 个不完整 episode 均为 `right_censored_identity_loss`，不是 administrative censor。来源原生 raw exclusion 另有独立计数。

## 可评分负暴露

负暴露由相邻帧的半开时间区间组成，两个端点都必须满足：

1. causal route 为 `known`；
2. route-relevant all-person truth 完整；
3. 不存在 `approaching_route` 或 `route_intersecting` active truth；
4. 时间戳严格递增且相邻间隔不超过 1 秒。

以 62,229 帧、41 条序列的全部相邻 ledger row 构成固定 candidate-pair universe，共 `62,188 = 62,229 - 41` 对；其中 3,801 对 eligible，58,387 对按 canonical primary reason 排除，最终合并为 836 个非重叠区间，共 `297,376,110,945ns`。逐来源为：

| 来源 | 严格负暴露 |
| --- | ---: |
| `crowdbot_0410_mds` | `1.899799min` |
| `crowdbot_1203_shared_control` | `1.364599min` |
| `lilocbench_dynamics_0_front` | `0.070259min` |
| `lilocbench_lt_changes_dynamics_0_front` | `1.621612min` |

首组 CrowdBot superseded 数据没有被重新授权进入负暴露分母；旧窗口或宽松口径不能用来补齐差额。

| pair primary exclusion | pair 数 |
| --- | ---: |
| source 未获 R2-L1 exposure 授权 | 22,840 |
| causal route endpoint unknown | 13,957 |
| route-relevant person truth endpoint 不完整（含跨 window 未绑定） | 14,720 |
| active truth role endpoint | 6,855 |
| frame ID 不连续 | 15 |
| 非正时间或相邻 gap 超过 1 秒 | 0 |

单个 pair 可以同时命中多个拒绝原因；上表使用冻结优先级给每个 ineligible pair 选择唯一 primary reason，因此合计严格等于 58,387。receipt 另保留 non-additive 的全部原因计数和合法正间隔时长。
其中 14 个 `timestamp_gap_exceeds_maximum` 与 `frame_id_not_consecutive` 同时出现，所以其 primary count 为 0、all-reason count 为 14；没有 `timestamp_nonpositive`。

## 哈希绑定与机器验证

冻结协议：`configs/ustrf_route_target_metric_eligibility_r2_l1.json`

- protocol SHA-256：`4ab0c5dd687f7c9a3b791795271e4ecf5f23c1bbea41d2561503e6ce72e196ac`
- eligibility mask SHA-256：`b7dd5cfacc6f14153900bfaf811f3e76a1e188d1064013823f717f615e528157`
- denominator receipt SHA-256：`3f356ca69eb50bd176210d01bd9deb69e35acca46764b14603d0bb155d7b82bd`
- validation receipt：`VALID`，18 项检查通过
- validator 变异测试：38/38 通过
- validation receipt 另绑定 core、materializer、validator 与 mutation-test 四个实现文件的 SHA-256

验证器从冻结 allowlist 重新打开并校验 11 个候选盲依赖，完整重建 mask 和 receipt，要求字节级规范 JSON 完全一致。它还独立检查：

- 没有读取或运行候选输出；
- 6,369 个 unit ID 唯一且每个恰有 8 个指标；
- 0/0 只能是 `not_evaluable`，不能 pass/fail；
- pre-clear 或无 clear 的事件不能进入 clearance；
- critical boolean 不能在缺少唯一连续 critical interval 时单独取得资格；
- repeat truth pool 不能冒充实际候选分母；
- 62,188 个负暴露 candidate pair 覆盖全部逐序列 frame adjacency，每对都有 eligibility/primary/all exclusion reason；合并区间为正、无重叠、整数纳秒可复算；
- 62,229 行显式 frame ledger 逐行固定 frame ID、capture timestamp 和 route validity；sequence mask hash 与 receipt frame-ledger hash 均可复算，下一任务无需猜测 mask membership；
- 分母、来源/来源族贡献比例、排除/删失计数与 terminal-clear observability 均可从 mask 重算；
- 高于 L1 的全部权限保持关闭。

本地机器证据位于 `artifacts.local/evidence/ustrf-route-target-metric-eligibility-r2-l1/`；该目录为忽略的可复算输出，不是仓库真源。

## 下一独立边界

下一任务只允许：

1. 绑定本轮 protocol、mask 和 receipt 的精确 SHA；
2. C1、C2、C3 各运行一次；
3. 只为已具备资格的指标生成探索 profile，对 `repeat` 和 `evidence_age` 先检查运行时完整性条件；
4. `event_recall`、`regeneration` 和 `false_alerts_per_minute` 保持 L0，不得输出通过/失败；
5. 不选胜者、不调阈值、不改分母、不开放 Android shadow 或更高权限。

现有 `run_crowdbot_holdout_candidates.py` 仍只遍历旧 accepted event，混用 repeat/regeneration，并以 pose age 代替 capture→consume evidence age；下一任务不能原样复用它作为 R2-L1 scorer，必须实现 receipt-aware、逐指标 fail-closed 的独立探索入口。

若后续选择先补 L0 缺口，只能按缺口矩阵补特定合同：event recall 补显式 deadline，regeneration 补同一人物 post-clear 身份连续性，false-alert exposure 补严格 route/person/time 合格负暴露；优先修复标注和审计机制，不扩大无目标下载。
