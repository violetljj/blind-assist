# USTRF current-input policy feasibility bound R0

日期：2026-07-24

终态：`CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID`

最大权限：`CURRENT_INPUT_POLICY_FAMILY_EMPIRICAL_FEASIBILITY_BOUND_ONLY`

## 结论

在冻结的 current-input monotone lease family 内，不存在同时达到 `33/33` supported-cell coverage 与经验负暴露点率 `<=0.50 token/min` 的上界。

本轮没有生成或选择候选 policy。求解只对一个共享、正整数的 active-relation 持续时长变量做完整断点包络；保留连续 2 帧、one-token-per-track/reset、route unknown / track unobserved / relation gap / reset 的 fail-closed 语义与 no-renewal。为避免有限 TTL 低估 coverage，上界计算乐观地忽略 nominal TTL，但不让 token 跨越任何 fail-closed 失效。任何有限 TTL 都只能减少 coverage，且不会改变 qualification timestamp 的冻结风险计数。

完整 41 条候选无关序列 / 62,229 帧形成 `7,542` 个 track-reset scope、`31,500` 个完整 qualification-duration activation interval 和 `29,424` 个确定性 frontier segment。36 个 candidate cell 先按候选无关的 source/sequence/event 去重为 12 个事件：11 个有 oracle-supported relation，1 个无 active relation；求解在 11 个事件上完成后再按 C1/C2/C3 机械映射回 33 个 supported cell。

结果为：

| 上界 | unique supported event | candidate cell |
| --- | ---: | ---: |
| 忽略 nominal TTL、只保留 fail-closed 的最大 coverage | `8/11` | `24/33` |
| 同时限制经验负 token `<=2`（即 `<=0.50/min`） | `2/11` | `6/33` |
| 冻结要求 | `11/11` | `33/33` |

因此 coverage 上界本身已经不足；加入风险门后差距更大。终态只能为 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE`。该终态不授权把资格改为 250ms、延长 TTL、启用 renewal、生成 successor policy 或连接 opener。

## 冻结 family 与反记忆约束

policy decision 只允许：

- track identity 作为 permutation-equivariant scope；
- per-track active route relation；
- route validity；
- reset；
- 相对 active-run 起点的 causal elapsed timestamp。

所有 scope 使用同一个单调判定 `support_duration >= shared duration`，并保留至少连续 2 帧。raw track ID 数值/顺序/hash、source/sequence、frame ID、绝对 timestamp、candidate、event/truth/oracle、future、clearance 均不得进入决策。因而结果对 track-ID 双射、source/sequence 重命名与逐序列 timestamp 平移保持不变，也不能用精确 epoch 或序列名建立 lookup table。

求解器没有写出达到局部最优的持续时长、TTL、activation map 或 witness；证书只保留 frontier segment 数、完整 frontier SHA 与可验证上界。

## 风险口径与证据限制

风险沿用父 policy gate 的经验事件：track-reset 首次 qualification timestamp 落入冻结半开负暴露 interval。总暴露为 `4.95626851575min`，所以 `0.50/min` 点率门最多容许 `floor(4.95626851575 × 0.50)=2` 个负 token。

本 R0 明确只判冻结数据上的经验点率可分性，不声称可信总体风险界。当前暴露低于父门的零事件 `5.9915min` floor；即使负事件为 0，一侧 95% Poisson UCB 仍约为 `0.604433/min`。两个 LILocBench source 的 sequence cluster 支持也仍不足。该限制没有被包装成 family `NOT_FEASIBLE` 的原因；本轮结构性终态来自 `24/33 < 33/33` 的乐观 coverage 上界。

## 收据与验证

- config SHA-256：`a809d5a5ba2299883f9d0f1ea98e8506b5304f1fe3b082746639e3bcd869de5f`
- bound certificate SHA-256：`f7a7962cebe1863893e8140d8c8857b5c6d95a327c2f6be971b8c41848b22108`
- terminal SHA-256：`2d5b3eda1bc047302d8ce2c2de3998821fb81dcb4cbbcfc66b5cb3f935803cf6`
- validation SHA-256：`7ac13bbb90a6e684d7512008f9027e01f43d16d28fb75a5580c4dda6cb6bc865`
- focused tests：10 tests OK；覆盖 track-ID alpha-renaming、timestamp 平移、route unknown/reset 隔离、半开负 interval、two-frame/no-renewal activation ranges、toy FEASIBLE/NOT_FEASIBLE、输出不含 threshold/policy/witness 与二元终态。
- validator：`VALID`；重新绑定父 policy/attribution SHA，从 41 条 allowed-input ledger 重建全部 scope、duration interval、candidate-independent event、负暴露与完整 frontier。
- canonical local evidence：`artifacts.local/evidence/ustrf-current-input-policy-feasibility-bound-r0/`

## 权限与下一独立边界

本轮没有修改 detector、T0、route、reset、C1–C3、truth、oracle、clearance 或任何阈值；没有生成 policy，没有启用 renewal，没有连接 opener，也没有开放 selection、L2/L3、Android shadow、H2、人体、独立行走或生产权限。

按冻结停止规则，下一独立边界不再继续调整资格时长、TTL 或 renewal。若继续，只能另行预注册一个新的、候选无关的因果判别信号；它必须增加当前 track/relation/route-validity/reset/timestamp family 中不存在的可分信息，并重新保留 `33/33` coverage 与完整风险门。
