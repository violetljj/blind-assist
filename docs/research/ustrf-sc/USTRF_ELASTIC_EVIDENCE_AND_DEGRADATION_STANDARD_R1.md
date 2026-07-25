# USTRF 弹性证据与降级标准 R1

状态：`CURRENT / MACHINE_READABLE_CONTRACT_BOUND`

适用范围：USTRF-SC 的公开数据、设备数据、模型代理、逐帧 observation、跨模态 join、指标 eligibility 与来源准入。

## 一、目的

严格的目的，是阻止错误结论和越权，不是要求现实数据没有任何缺失。

未来所有数据判定必须先回答：

1. 当前要支持的具体 claim 是什么？
2. 该 claim 真正必需的字段和模态是什么？
3. 缺陷影响的是字段、对象、帧、窗口、sequence、source，还是所有证据？
4. 未受影响的部分还能合法支持什么较窄的 claim？
5. 缺失是否会产生系统性偏差；若会，应怎样限制外推？

只有证明缺陷污染了整个 claim 分母或破坏不可变来源身份，才能把局部问题升级成全局关闭。

## 二、五类问题必须分开

| 类别 | 例子 | 默认处理 | 不得做 |
| --- | --- | --- | --- |
| 结构/完整性损坏 | hash/CRC 漂移、时钟倒退、frame chain 不闭合、同一 join key 多义、跨 reset 泄漏 | 关闭最小被污染单元；若污染无法定位才升级 sequence/source | 用下游结果猜测或修补 source fact |
| 正常观测缺失 | FOV 差异、遮挡、2D-only、3D-only、某对象暂时未观测 | 对受影响对象/帧/指标 `abstain`；保留其他合法证据 | 要求所有模态永久 100% 共现 |
| 支持度不足 | 分母太小、覆盖不均、某来源/场景稀缺 | `insufficient_support` / `not_evaluable`；缩小 claim scope | 写成算法失败或数据全无价值 |
| 性能失败 | recall、误报、延迟或质量指标未过冻结门 | 拒绝该候选、方法或 claim | 反向宣布 source transport 不可用 |
| 权限/外推缺口 | 无 route truth、event truth、人体或生产证据 | 保持对应 authority `HOLD/CLOSED` | 抹掉已经成立的开发、诊断或几何证据 |

这些类别不得互相替代。特别是：

- 数据存在正常缺失，不等于结构损坏；
- 方法性能不足，不等于来源不可用；
- selection/production 权限关闭，不等于开发/诊断证据必须删除；
- `not_evaluable` 不等于零，也不等于失败。

## 三、三轴报告，禁止单一 terminal 吞掉信息

每次结果必须分开输出：

1. `artifact_integrity`：`VALID` 或 `INVALID`，只回答 immutable source/packet 是否可复算；
2. `claims[]`：每个 claim 独立判定；
3. `authority_ceiling`：当前最高只能用于 diagnostic、exploratory、selection、shadow 或 production 的哪一层。

claim 状态按以下优先级判定：

1. `INVALID_DEPENDENCY`
2. `NOT_EVALUABLE_AUTHORITY_MISSING`
3. `NOT_EVALUABLE_INSUFFICIENT_SUPPORT`
4. `AVAILABLE_WITH_DEGRADATION`
5. `AVAILABLE_COMPLETE`

这里的 `AVAILABLE` 只表示可以在声明分母和范围内计算或研究，不表示方法性能通过，更不授予 selection、Android、人体或生产权限。

整体状态只允许：

- `VALID`：artifact 可复算且全部声明 claim complete；
- `VALID_WITH_PARTIAL_OR_DEGRADED_CLAIMS`：artifact 可复算，但至少一个 claim 降级或不可评；
- `INVALID_GLOBAL_INTEGRITY`：污染无法局部化，artifact 本身不可作为证据。

一个 claim 的 `not_evaluable` 不得覆盖另一个独立 claim 已成立的 availability。

## 四、最小影响粒度原则

处置粒度按以下顺序选择：

```text
field → observation/object → frame → window → sequence → source → program
```

每次只能选择能够完整隔离缺陷的最小粒度。升级到更大粒度必须在 receipt 中给出：

- `propagation_evidence`：为什么更小隔离不能阻止污染；
- `affected_denominator`：受影响和未受影响的精确数量；
- `claim_dependency`：该缺陷为何是当前 claim 的必要输入；
- `counterfactual_subset`：未受影响子集为何仍不足，或可支持什么较窄结论。

缺少这些字段时，不得使用 sequence/source/program 级 hard fail。

## 五、claim-specific eligibility

不存在跨任务通用的“完整率必须 100%”。每个 claim 在查看 outcome 前声明：

- `required_roles`：缺少即该 unit 不可评；
- `optional_roles`：缺少只降低辅助分析；
- `unit_of_analysis`；
- `join_key` 与唯一性条件；
- `maximum_gap/interpolation`；
- `minimum_support`；
- `missingness_risk` 与分层报告；
- `maximum_claim_scope`。

示例：

| Claim | 必需 | 可选或另立指标 | 合法缺失处理 |
| --- | --- | --- | --- |
| 3D-native person trajectory availability | 3D `label_id`、3D center、source time、bounded pose、静态 frame chain | 2D bbox、RGB projection | 3D-only 仍可进入 3D trajectory；cross-modal 指标 abstain |
| 2D↔3D projection consistency | 同帧唯一 2D/3D `label_id`、相机/几何变换 | motion | 2D-only/3D-only 从该指标排除并进入 union denominator 缺失账 |
| route-conditioned event | route truth、event truth、对象/时间/几何 | 非必要辅助模态 | 缺 route/event 的 unit 不进入该 claim，但可保留几何/跟踪证据 |
| production safety | 独立设备、人体、风险与回归 authority | 开发 proxy | 任一缺口保持 production closed，不改写较低层证据 |

## 六、覆盖率不是隐藏分母

每个适用指标必须同时报告：

- union denominator；
- eligible denominator；
- excluded / abstained 数量；
- coverage；
- 按 source、sequence、scene、visibility、missingness reason 的分布；
- 性能只在 eligible denominator 上计算；
- 权限判断同时检查 coverage、偏差和最差 cluster。

描述性 coverage band：

- `HIGH_COVERAGE`：`>=95%`；
- `MODERATE_COVERAGE`：`80%–<95%`；
- `LOW_COVERAGE`：`<80%`。

band 本身不是 pass/fail。任务可以依据 claim 风险另冻最低支持度，但不得在看见结果后把 band 改成晋级门，也不得用 pooled 高覆盖掩盖关键 cluster 为 0。

每个 claim 必须满足：

```text
expected_denominator
= eligible_denominator
+ abstained_denominator
+ invalid_denominator
```

`expected_denominator` 来自 source-native universe，不能改成交集。imputed/interpolated record 必须单列 provenance，不得增加“直接观测”计数。

## 七、允许的处置

1. `ADMIT_COMPLETE`：所有 required roles 对该 claim 完整；
2. `ADMIT_WITH_ABSTENTION`：结构有效，局部缺失已隔离，coverage/偏差显式；
3. `DIAGNOSTIC_ONLY`：支持度或代表性不足，但仍能验证运输、几何、机制或回归；
4. `NOT_EVALUABLE_FOR_CLAIM`：当前 claim 必需角色整体不足；
5. `REJECT_METHOD_OR_CANDIDATE`：数据可评，但方法性能未过门；
6. `FAIL_CLOSED_CORRUPT_EVIDENCE`：不可定位或不可隔离的结构/身份/时间/provenance 损坏；
7. `AUTHORITY_HOLD`：技术证据存在，但 selection、route、human 或 production 权限不足。

同一 source 可以对不同 claim 同时具有不同 disposition。

## 八、反钻空子规则

- 历史 terminal 永久保留；语义修正必须新版本，不追溯改写；
- 不得因放宽处置粒度而降低时钟、身份唯一性、hash、frame chain、插值上界或 reset 门；
- 不得删除缺失 unit、改 union denominator、把 missing 记成成功/失败/零；
- 不得在 outcome 暴露后调 eligibility 以救候选；
- 缺失若与距离、遮挡、场景或难例相关，必须标为 informative missingness，并限制 claim；
- 局部数据通过不能自动增加 selection、Android、human 或 production authority；
- validator 必须从 immutable source/packet 重建 eligibility 与分母，不能只信 receipt；
- 任何 sequence/source/program 级关闭都必须通过“最小粒度无法隔离”的升级证明。
- 缺失原因只有在 source fact 支持时才能写成 FOV/occlusion；否则必须是 `unknown_missing`；
- 每个 defect 必须记录 class、scope、受影响模态/claim、localized、denominator impact 和 evidence refs；
- eligibility 不能由下游 motion、risk 或候选 outcome 反向决定。

## 九、具体问题具体分析记录

每次审计必须持久化：

```text
claim_id
required_roles / optional_roles
unit_of_analysis
defect_class
missingness_mechanism
affected_unit_ids
union / eligible / abstained denominators
coverage and cluster distribution
propagation_evidence
bias_risk
disposition
maximum_claim_scope
authority_granted / authority_closed
```

如果 `missingness_mechanism` 未知，受影响 unit 先 abstain；未知本身不自动污染所有未受影响 unit。

## 十、与既有标准的关系

本标准把 Evidence Maturity V2 已有的“逐指标 eligibility、删失、空分母 `not_evaluable`、局部证据保留”推广到 source、frame、object 和 cross-modal join。V2 的统计分母、hard veto、candidate-blind 与权限分层继续有效。

本标准不自动重开任何历史任务。怀疑被误杀的结果必须逐项审计：

1. 保留旧 terminal；
2. 判断旧失败属于哪一类和哪一粒度；
3. 另立版本化 recovery；
4. 只继承 immutable source facts；
5. 重新冻结 claim-specific eligibility 后再运行。
