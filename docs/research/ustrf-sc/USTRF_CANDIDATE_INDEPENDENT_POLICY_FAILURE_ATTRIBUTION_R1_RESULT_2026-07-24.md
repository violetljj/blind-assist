# USTRF candidate-independent policy failure attribution R1

日期：2026-07-24

终态：`POLICY_FAILURE_ATTRIBUTION_CLOSED / VALID`

最大权限：`CANDIDATE_INDEPENDENT_POLICY_FAILURE_ATTRIBUTION_ONLY`

## 结论

冻结 policy 的 coverage failure 已闭合归因，但没有被修复。24 个 supported-cell miss 中：

- 12 个 cell 只有资格不足；
- 6 个 cell 只有 oracle 落在 500ms TTL 之后；
- 3 个 cell 同时含资格不足与 relation gap 提前失效；
- 3 个 cell 同时含资格不足与 route unknown 提前失效。

因此不能把 24 个 cell 强行压成三个互斥机制而丢掉混合原因。本轮以 96 次 oracle qualification opportunity 为互斥原子，每次严格归一类：资格不足 `39`、TTL 后 oracle `39`、relation gap 提前失效 `12`、route unknown 提前失效 `6`、track unobserved 提前失效 `0`，unexplained `0`。24 个 cell 仍全部闭合，其中 6 个显式保留 `mixed=true`。

这说明 coverage reject 不是单一 TTL 问题：一半 opportunity 在 oracle 到来时尚未形成冻结的 `2 frames + 500ms` 资格；另一半中既有 TTL 到期，也有 relation/route 更早失效。该结果只解释当前 policy，不能据此直接放宽 TTL、降低资格门或选择新 policy。

## 归因合同

本轮只读取并复算上一阶段已冻结的 41 条 policy ledger、1,448 个 token、policy terminal/risk ledger 与既有 oracle ledger。没有重跑 detector、T0、route、reset、C1–C3、truth 或 oracle。

每次 oracle qualification 使用同一 `source/sequence/reset/track` 联结 policy token，并按 timestamp 检查半开有效期：

`qualification_timestamp_ns <= oracle_timestamp_ns < effective_valid_until_timestamp_ns`

分类保持以下差异：

- `QUALIFICATION_INSUFFICIENT`：该 track/reset 从未形成 token，或 oracle 早于后续 policy qualification；后来的 token 不回写旧时点覆盖。
- `ORACLE_AFTER_TTL`：token 已形成，但 oracle timestamp 已到达或越过 nominal/effective TTL。
- `ROUTE_UNKNOWN_BEFORE_ORACLE`、`TRACK_UNOBSERVED_BEFORE_ORACLE`、`RELATION_GAP_BEFORE_ORACLE`：继承 producer 已记录的首次 fail-closed 失效原因。
- reset、sequence end 与未知原因保留独立残差类，不硬塞进上述原因；本次计数均为 0。

对资格不足 opportunity，ledger 另存 oracle 时点的 route-known、track-observed、active-relation、连续支持帧数、持续时间和后续 token 是否出现，避免把“没有 token”误读为“完全没有 relation 证据”。

## 负暴露 token

34 个负暴露 token 全部按 `token_id` 唯一联结到既有 invalidation，unattributed `0`。失效原因：

| 失效原因 | token |
|---|---:|
| `ttl_elapsed` | 16 |
| `active_relation_gap` | 9 |
| `track_unobserved` | 8 |
| `route_unknown` | 1 |

ledger 同时按 source、sequence、invalidation reason 输出 23 个非空分组。该账本解释负暴露 token 的生命周期终点，不把它与 supported-cell miss taxonomy 混为同一分母，也不改变上一阶段 `34/4.956min=6.86/min` 的风险拒绝事实。

## 可复现证据

- config SHA-256：`b838192e48e1d01f3f4cc75c19388dd7c5dc771e39b4de6c2fcc57cbe0942e21`
- supported-miss ledger SHA-256：`cf4c7b12ee59b1270010861498f04de36966b9b4599567cc400fb1b6dc7eb05a`
- negative-token ledger SHA-256：`246dc3726803a9f6d0bd5fd3ee1b85391a9bc6f2844ec54ff9a30c3d519a61ed`
- terminal SHA-256：`d9ef42df1844913e43b6d2275cd7907cf3ac4701d292bfd52b871524e5cd28c5`
- validation SHA-256：`26e6cce3ae17e0353bdd2f86f75cf8b863f5bd32b1e189d122aa9c47ef755b69`
- focused tests：10 tests OK，覆盖资格不足两种机制、timestamp 半开 TTL 边界、route/track/relation 分离、reset/sequence 残差、未知原因 fail-closed，以及负 token 的 source/sequence/reason 联结。
- validator：`VALID`；重新复算父 policy terminal/risk、96 次 opportunity、24 个 cell 的 mixed signature、34 个负 token 与全部汇总。

## 权限与下一边界

本轮没有生成、比较或授权任何 successor policy；没有连接 opener，也没有开放 selection、L2/L3、Android shadow、H2、人体、独立行走或生产权限。上一阶段 `POLICY_COVERAGE_REJECT` 保持不变。

若继续，必须另行冻结一个且仅一个 policy 变量，并同时保留 coverage 与完整负暴露风险门；本归因结果本身不授权选择变量或修改阈值。
