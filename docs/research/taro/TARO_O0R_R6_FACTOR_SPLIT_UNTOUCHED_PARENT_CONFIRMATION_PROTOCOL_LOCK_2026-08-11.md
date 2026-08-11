# TARO O0R R6 factor-split untouched-parent confirmation protocol lock

状态：`PROTOCOL_FROZEN / IMPLEMENTATION_ALLOWED / EXECUTION_NOT_AUTHORIZED`

机器合同：[JSON](TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_PARENT_CONFIRMATION_PROTOCOL_LOCK_2026-08-11.json)

## 冻结算法

R6 固定为 `R5_SELECTED_SUPPORT_BOUNDARY_PLUS_ALWAYS_R1_QUERY_CLEARANCE_V1`：

- SUPPORT、BOUNDARY 和 extraction status 从 Phase A 已选择的 source-only component 原样复制；
- QUERY_CLEARANCE 始终从 R1 baseline component 原样复制；
- 三个 factor 各自携带 depth SHA-256，不再把不同 ownership 压进一个虚假的单 depth lineage；
- 零学习参数、零新阈值、零 outcome-dependent reselection。

该算法来自 R5 正式 FAIL 后的 post-hoc factor interaction，因此现有 24 个 ARKitScenes Training parents
只能用于 formation/canary，不能用于 R6 confirmation。

## 未触碰确认前门

未来 confirmation 至少需要 8 个 parent，且必须在任何 model/truth read 前冻结 exact roster、frame counts、
source/truth hashes 和 data-use authority。parent 必须与 R4/R5/R6 formation 的全部 24 个 Training parents
不相交。当前没有该数据锁，所以 execution=false。

Phase A 必须先封存 SUPPORT/BOUNDARY owner 和 `QUERY_CLEARANCE=R1_BASELINE`；Phase B 才能打开 FARO，
生成 baseline/selected components，再由确定性 compositor exact-copy 三个 factor。FARO、task metric、knownness
或 prior outcome 都不能改变 owner。

## 冻结门槛

沿用 R5 的 parent-macro height/normal、8-parent denominator、extraction/query no-regret，并新增
`BOUNDARY_EVALUABILITY_NO_REGRET`。任一 parent 缺少 paired support metric 时结果为 NOT_EVALUABLE；UNKNOWN
永远不是 negative。

## 下一动作

当前唯一允许动作是 `TARO_O0R_R6_FACTOR_SPLIT_IMPLEMENTATION_LOCK`：实现 roster-independent component
schema、factor-depth lineage、确定性 compositor、validator 和 mutation tests，不读取新数据、不执行模型或
truth scoring。

真正执行还必须另有 `TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK`。即使未来 PASS，claim ceiling 也只到
WILD_LAB parent-disjoint factor-level task-metric confirmation，不涉及 formal O0R、部署、产品或安全。
