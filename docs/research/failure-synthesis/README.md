# BlindAssist causal failure diagnosis

状态：`current / D_ORACLE_1_UNIQUE_P0 / UNIQUE_P0 / PROTOCOL_FROZEN / BLOCKED_ON_SOURCE_ACTION_TRUTH_POLICY_LOCK / NO_OUTCOME_ACCESS / NO_EXECUTION / NO_SEARCH / DEFAULT_APP_UNCHANGED`

## 当前主张

D-ORACLE-1 不研发新模型。它只用一个 fresh parent-disjoint cohort 和冻结的三臂 matched causal
ladder，判断 BlindAssist 的主要可恢复损失位于 downstream target-policy stack，还是 estimated
representation：

```text
A  Direct Action Oracle
B  Perfect Source Geometry -> Current Policy
C  Estimated Representation -> Same Current Policy
```

B/C 后的 policy implementation、config、threshold、coverage rule、evaluator 和 parent denominator
必须逐 hash 相同。parent-local geometry permutation 只作机制 control，不是第四个竞争臂。

## 当前结论

- [D-ORACLE-1 protocol](D_ORACLE_1_MATCHED_CAUSAL_LADDER_PROTOCOL_2026-08-17.md)及
  [machine contract](D_ORACLE_1_MATCHED_CAUSAL_LADDER_PROTOCOL_2026-08-17.json)已冻结；
- 现在没有 source roster、action oracle ledger、event-evaluation ledger、B source-geometry ledger、
  C estimated-representation identity、policy/config hash 或 outcome；
- `U` 是 parent-level event utility，只用于 gap attribution；hard gates、native/matched coverage、
  false-clear、false-block、transition 与 worst-parent 不得被 `U` 抵消；
- H3/H4 不在本轮拆分。只有 `A materially > B` 后，才允许另立小型 D-ORACLE-2；本协议不提前
  增加 `Perfect Task Target -> Current Policy` 或 `Current Target -> Oracle Policy`；
- Failure Synthesis 的 H3/H4/H2/H1 排序是待检验 prior，不约束结果。若 `A≈B≫C`，必须降低 H3、
  提升 H2，不得修改本协议救原结论。

## 唯一 successor

`D_ORACLE_1_SOURCE_ACTION_TRUTH_POLICY_LOCK`：在任何 fresh RGB、geometry、action label、event truth、
model output 或 arm metric 打开前，一次性冻结 exact roster、两个独立 truth roles、B/C common policy
implementation/config/evaluator hashes、C 的既有 frozen representation identity、permutation map 与执行 root。

该 successor 只授权 lock/preflight，不授权三臂 execution 或 outcome access。lock validator PASS 后仍需
显式 activation 才能运行一次 matched ladder。

## 当前允许

- 只读 source/metadata capability audit，不读取候选输出或 action/event outcome；
- 建立 schema、synthetic canary、hash/identity validator 和 source-role conflict check；
- 复核 protocol 与 Failure Synthesis 的一致性。

## 当前禁止

- 训练或修改 encoder、depth、temporal、selector、policy、threshold、feature contract 或 evaluator；
- 执行 A/B/C、读取 fresh outcome、按结果换 parent/denominator/gate；
- 把 permutation control当第四竞争臂或用它选择 B/C 参数；
- 提前执行或设计 D-ORACLE-2 arms；
- 恢复 SVRF index/materialization、TARO、Q-Plane、B1/A0、obstacle router 或其他 representation search；
- 修改默认 App，或产生产品、安全、真实用户有效性结论。

## Claim ceiling

当前只证明因果诊断问题、三臂身份、gap formula、统计单位、机制 control 和停止条件已预先冻结。
没有科学 outcome，也没有证明 H1/H2/H3/H4 任一根因。默认 App 不变。
