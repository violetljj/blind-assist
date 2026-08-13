# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / R10_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R10_NO_PROMOTION / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_PHASE_A_INDEPENDENT_VALIDATION_PASS / R11_SOURCE_ONLY_TOP24_ONE_SHOT_CONSUMED_PASS / R11_TOP24_INDEPENDENT_VALIDATION_PASS / R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_PASS_NON_EXECUTING / R11_FARO_PHASE_B_EXECUTION_LOCK_AUTHORIZED_UNCONSUMED_PARKED_NONBLOCKING / TASK_OBSERVABILITY_PAIR_SUPPORT_R7_R10_ZERO_1S_PAIRS / TASK_OBSERVABILITY_D0_ACTIVE / R11_SCIENTIFIC_NOT_RUN / DEFAULT_APP_UNCHANGED`

本页只维护 TARO 当前状态、权限和唯一算法 successor。较早完整 R0–R11 叙事保存在
[14d8ad7e 历史快照](archive/README_FULL_HISTORY_2026-08-13.md)，不能从中恢复旧权限。

## 当前主张

TARO 是独立并行 `WILD_LAB`：在声明的米制锚和冻结 factor/reducer 下，用低维 residual
gauge posterior、可观测子空间和同预算额外观测，让 body/path-specific query 先于完整场景
达到局部可识别。`UNKNOWN`、缺字段和不可观测方向永不转成 negative。

当前 Development 只检验 positive-occupancy task-directed observability，不输出 `CLEAR`，也不要求
用户迈步取证。TARO 与 [Assistive Geometry](../assistive-geometry/README.md) 并列，不从 DepthART、
Android、HTP 或默认 App 自动继承权限。

## 当前结论

- R10 以 dual-class coverage `NOT_EVALUABLE` 收口；不得改 selector、threshold、denominator 或 gate 回救。
- R11 exact-48 source-first Phase A、source-only 48→24 selection 与各自独立验证均 PASS；selected 24
  identities 已不可变封存，unselected FARO 仍为 0。
- selected-only FARO Phase-B implementation lock 已完成；one-shot execution lock 已冻结为
  `AUTHORIZED_UNCONSUMED`，正式 FARO 尚未读取。该 Formal arm 当前 parked/nonblocking，权限没有被
  改写，也不是本轮 Development 的前置条件。
- source-only pair-support audit 显示：R7 的 170 帧与 R10 的 710 帧虽然 pose 完整，但最小相邻间隔
  都是 2 秒；冻结的 1 秒窗口内 pose-valid adjacent pair 均为 0。它只证明当前 cohort 无法评价
  该机制，不证明时序或主动观测无效。

## 当前证据入口

- [R10 terminal](TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_RESULT_2026-08-12.md)
- [R11 Phase-A independent validation](TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATION_RESULT_2026-08-13.json)
- [Top-24 result](TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_RESULT_2026-08-13.json)
- [FARO Phase-B implementation lock](TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_2026-08-13.md)
- [FARO Phase-B execution lock](TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json)
- [Pair-support audit](TARO_TASK_DIRECTED_OBSERVABILITY_PAIR_SUPPORT_AUDIT_RESULT_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [TARO Module](../../../scripts/research/taro/README.md)

## 唯一 successor

`TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_R0`：

1. 先只按 metadata/pose/timestamp，在已披露、pose-rich 的 `PROJECT_CONSUMED_DEVELOPMENT` source 中
   验签合法 pair 能力；source 选择不得读取 task outcome；
2. 用相同“一次额外观测”预算比较 static R7、最佳合法 passive neighbor、固定 micro-baseline、
   generic max-parallax 与 task-directed oracle；micro arm 只离线使用自然存在的 `6±2 cm` translation、
   `≤5°` rotation pair，不要求真实用户行动；
3. 只允许 truth-consistent `UNKNOWN→OCCUPIED`；既有正确 known 必须保留，并按 parent macro 报告
   recovery、false-occupied、UNKNOWN、额外帧数、时延与位姿成本；
4. 新 source 仍无合法 pair 就记 `NOT_EVALUABLE_DATA_OBSERVABILITY`；passive 不优于 static 停时序，
   micro 不优于 passive 停主动支线；只有 task-directed 优于同预算 passive 与 generic 才研究 learned scorer。

## 当前允许

- 只读 candidate-input JSON 做 source capability audit；
- 在隔离输出根上执行上述 positive-occupancy-only Development canary；
- 重放 hash-bound tests、validator 和只读 evidence 复核；
- 保留 Formal lock 的 `AUTHORIZED_UNCONSUMED` 状态，不消费、不伪装为撤销或科学结果。

## 当前禁止

- 在 R7/R10 的 1 秒合法 pair 为 0 后训练时序模型或事后放宽窗口；
- 用不同额外帧预算比较 sensing arms，或只报告 recovery 而隐藏 false-occupied/known retention/cost；
- 在 Development canary 中输出 `CLEAR`、把 UNKNOWN 当 negative、读取 R11 protected FARO outcome，
  或修改 sealed selection/selector/candidate/threshold；
- 覆盖、resume、删除或重跑已消费 one-shot，或越级训练、Android/QNN/HTP、默认 App、产品或安全结论。

## Claim ceiling

当前只证明 source-first Phase A、source-only top-24 selection、FARO Phase-B implementation/execution-lock
准备，以及 R7/R10 的 1 秒 pair-support 缺口；不证明 FARO outcome、task-directed sensing 有效、fresh
dual-class confirmation、移动端可行性、产品有效性或用户安全。
