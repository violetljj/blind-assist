# D-ORACLE-1 matched causal ladder protocol

状态：`FROZEN_PREOUTCOME / UNIQUE_P0 / BLOCKED_ON_SOURCE_ACTION_TRUTH_POLICY_LOCK / NO_EXECUTION`

机器合同：
[D_ORACLE_1_MATCHED_CAUSAL_LADDER_PROTOCOL_2026-08-17.json](D_ORACLE_1_MATCHED_CAUSAL_LADDER_PROTOCOL_2026-08-17.json)

## 1. Question

BlindAssist 是否长期优化错层：把 estimated representation 替换为 perfect source geometry 后，current
policy 能否接近 direct action oracle 的 task utility？

本轮只分离：

```text
downstream target-policy stack loss
vs
estimated representation loss
```

`A->B` 混合 H3 target/supervision 与 H4 policy/objective；本轮不拆 H3/H4。

## 2. Three competitive arms

| Arm | Input | Frozen downstream |
|---|---|---|
| `A_DIRECT_ACTION_ORACLE` | output-blind frame/action oracle ledger：`ALERT / HOLD / CLEAR / UNKNOWN` | common event state machine and evaluator adapter only；不经过geometry policy |
| `B_PERFECT_SOURCE_GEOMETRY_CURRENT_POLICY` | source-native metric depth、pose/ground、obstacle/traversability与显式UNKNOWN | exact current policy implementation/config/threshold/evaluator hash |
| `C_ESTIMATED_REPRESENTATION_CURRENT_POLICY` | activation前选定的既有 frozen estimated representation；禁止新训练/搜索 | 与B逐hash相同 |

B/C 必须使用相同的 feature contract、policy binary/source、config、threshold、state machine、coverage rule、
missing-data rule、evaluator和parent denominator。B不得因perfect geometry获得额外参数、校准或policy branch。

## 3. Truth-role isolation

为避免 A 自评分，必须在 arm output 前冻结两个独立角色：

1. `ACTION_ORACLE_LEDGER`：逐时刻给出 direct action与UNKNOWN；生产者不能读取B/C输出、policy score或
   event evaluator aggregate。
2. `EVENT_EVALUATION_LEDGER`：独立给出parent type、positive/negative、required alert interval、timely
   interval、critical miss、pass/clear interval与不可评估原因；生产者不能读取A/B/C输出。

同一观察可由不同人员/流程复核，但两个ledger必须独立生成、独立hash、在arms前封存。AI review若参与必须
保留provenance，不能冒充human/source-native truth。任何required parent的冲突或缺失保持UNKNOWN，并使
对应预冻结gate按合同fail-closed；不得多数票强制变negative。

## 4. Fresh matched cohort

source/action lock必须一次性冻结至少12个从未被BlindAssist outcome访问或用于训练/选择的parents：

- positive：至少6个；`BLOCKING_OBSTACLE / LEVEL_CHANGE_OR_DROP / DYNAMIC_APPROACH` 各至少2个；
- negative：至少6个；`PARALLEL_NONINTERSECTING / NORMAL_WALKABLE / NONAPPROACH_DYNAMIC` 各至少2个；
- 每个positive parent必须含pre-event、required-alert与pass/clear phase；
- 每个parent必须同时具备A action oracle、B完整source geometry、C输入与独立event truth资格；
- source、session、parent与ancestry对历史train/dev/selection/confirmation逐项查重。

若无法满足exact roster，不缩分母、不换较弱source、不执行；终态为 `NOT_EVALUABLE_COHORT`。

## 5. Utility and hard gates

### Parent utility

`U` 只用于因果gap归因，不决定promotion。每个parent先得到适用的event clauses：

- positive：`event_hit`、`timely_hit`、`post_pass_clear`；
- negative：`no_false_alert_event`；
- required UNKNOWN/missing clause不可zero-fill。

`u_parent` 是该parent全部适用binary clauses的算术平均。先分别计算positive-parent macro与
negative-parent macro，再等权平均为 `U`；frame从不作为独立统计单位。

同时必须报告：

- `U_native`：每arm全部native-evaluable locked parents；
- `U_matched`：A/B/C全部evaluable的exact common parents与common time support；
- positive/negative macro、paired parent delta、median、worst-parent；
- event hit、timely、critical miss、false-alert event、clear；
- policy-native known coverage、matched coverage、false-clear、false-block、UNKNOWN、transition；
- 所有pooled/frame metrics仅作diagnostic。

gap只用 `U_matched`：

```text
G_downstream     = U(A) - U(B)
G_representation = U(B) - U(C)
G_total          = U(A) - U(C)
```

当且仅当 `G_total >= 0.10`、三个U finite且两个component gaps均非负时，报告：

```text
R_downstream     = G_downstream / G_total
R_representation = G_representation / G_total
```

否则仍报告raw gaps，但ratio标记 `NOT_INTERPRETABLE_AS_ATTRIBUTION_PROPORTION`。

### Frozen gates

本诊断的action/event前门：

- native与matched parent coverage均 `>=0.90`；
- A 的 `U_matched >=0.80`，positive timely-hit `>=0.80`，positive clear `>=0.80`；
- A critical misses `=0`；negative false-alert event rate `<=0.20`；
- 任一required parent metric必须finite；worst-parent只报告，不被macro隐藏。

B/C还共同继承current policy quality gates；activation必须绑定exact config hash，至少包含：known coverage、
clearance coverage/MAE、false-clear、false-block、UNKNOWN、temporal clearance delta、transition agreement与
worst-parent false-clear。activation不得更改既有阈值来适配B。

任何hard gate失败均不能由U、gap、ratio或bootstrap CI覆盖。

## 6. Statistics

- parent是唯一resampling unit；frame/time slot只是parent内纵向观察；
- paired parent deltas按exact common parent计算；
- 报mean、median、worst-parent与每parent表；
- 固定seed `20260817`，按positive/negative strata做10,000次paired parent bootstrap；同一resample indices
  同时用于A/B/C；报告U、raw gaps和可解释ratio的95% percentile CI；
- 不报告frame-level p-value，不把bootstrap CI当产品/安全确认。

预声明gap判读：

- `MATERIAL_GAP`：absolute gap `>=0.10` 且paired bootstrap 95% CI lower bound `>0`；
- `PRACTICAL_EQUIVALENCE`：absolute gap `<=0.05` 且95% CI完全位于 `[-0.10,+0.10]`；
- 其余为 `INDETERMINATE`，不得硬塞进四种故事。

## 7. Mechanism control

`K_B_PARENT_LOCAL_DERANGEMENT` 不是第四竞争arm。它将每个parent的B geometry packets按timestamp排序，
以固定half-cycle derangement映射到同parent其他时刻；保持parent、packet count、native coverage、marginal
geometry distribution和同一policy hash，只破坏time/action alignment。n不足或无法无fixed-point时该parent
control为UNKNOWN，不替换其他parent。

报告 `G_geometry_use = U(B)-U(K_B)`：

- `POLICY_GEOMETRY_USE_SUPPORTED`：delta `>=0.05`、paired median `>0`、至少2/3 evaluable parents为正；
- `POLICY_GEOMETRY_USE_NEAR_IDENTICAL`：absolute delta `<=0.02` 且bootstrap CI包含0；
- 其他为 `POLICY_GEOMETRY_USE_INDETERMINATE`。

control不得进入A/B/C gap、ratio、winner或参数选择。如果B与control近似，只支持“当前policy未证明有效利用
geometry”，不证明geometry target无价值。

## 8. Outcome map

| Frozen pattern | Interpretation | Successor |
|---|---|---|
| `A materially > B`, `B equivalent C` | downstream target-policy stack主导；representation perfection未救当前policy | 停止representation search；另立小型D-ORACLE-2拆H3/H4 |
| `A equivalent B`, `B materially > C` | 当前target-policy stack有headroom，representation loss主导 | H2升为primary；才允许新representation protocol |
| `A materially > B materially > C` | downstream与representation均有损失 | 按raw gap与ratio优先较大层；不得同时开搜索 |
| A action/event gate fail | direct action construct/evaluator本身不成立 | 上溯task truth/evaluator/observability；不运行新模型 |
| 任一gap indeterminate | 当前cohort不能区分 | 保留prior，不调门、不补parent救结论 |

若结果反转Failure Synthesis prior，必须更新root ranking；禁止结果后修改protocol、U、equivalence margin、
parent roster、policy或failure history。

## 9. Pre-execution bindings

在任何outcome access前，activation必须hash-bind：

- exact parent/source/session/ancestry roster与prior-use audit；
- action oracle ledger schema/producer/receipt；
- event evaluation ledger schema/producer/receipt；
- B source-geometry schema、units、coordinate frames、UNKNOWN policy与ledger；
- C existing representation model/checkpoint/code/preprocess identity；
- B/C common feature contract、policy implementation、config、threshold与evaluator；
- A adapter与common event state machine；
- matched/native coverage calculator、bootstrap implementation与permutation map；
- immutable output root、validator与failure-state journaling。

目前这些bindings均未建立，因此protocol frozen但execution未授权。

## 10. Prohibitions and stop rules

- no training, search, retuning, policy branch, threshold change or new representation；
- no D-ORACLE-2 arm in this protocol；
- no source/parent replacement after any action/event/geometry/model outcome access；
- no oracle-only higher coverage masquerading as gain；native与matched必须同时报告；
- no UNKNOWN-as-negative、zero fill、missing-parent drop或frame-level significance inflation；
- no change to frozen historical terminals regardless of result；
- no Android/default-App/product/safety claim。

唯一下一动作是 `D_ORACLE_1_SOURCE_ACTION_TRUTH_POLICY_LOCK`；其validator PASS前不执行任何arm。
