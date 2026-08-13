# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / R11_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R12_THREE_SOURCE_LABEL_SATURATION_LOCALIZED / R13_TASK_EVIDENCE_ORACLE_HEADROOM_PASS / POSE_SCORER_NOT_YET_RUN / DEFAULT_APP_UNCHANGED`

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
  identities 已不可变封存。
- selected-only FARO Phase B 已正式消费并通过独立复算：674/674 frames、6,066 queries、678-file root，
  unselected FARO=0。正式终态为 `NOT_EVALUABLE_DUAL_CLASS_COVERAGE`：28 个 definite-CLEAR queries
  覆盖 7 parents，却只来自 10 physical frames，低于冻结的 12；10/10 clear-frame specificity 的单侧
  95% Wilson 下界为 `0.787058`，低于 `0.8`。
- R11 与 R7 在所有 definite labels 上完全相同，只额外把 1 个 truth-UNKNOWN query 从 `OCCUPIED`
  变为 `UNKNOWN`；weak-distal abstention 在 fresh cohort 没有产生预期的 clear-negative-control 效果。
- 单变量 post-hoc Development replay 只把 R9 `far_fraction_index` 从 0 改为 2：clear-frame recall
  从 `0.60` 升至 `1.00`，eligible frames 从 34 增至 64、未超过冻结的 2× 上限 68；precision 从
  `17.65%` 降至 `15.63%`。该候选保留为高召回 ranking proxy，不是 CLEAR classifier 或 confirmation。
- source-only pair-support audit 显示：R7 的 170 帧与 R10 的 710 帧虽然 pose 完整，但最小相邻间隔
  都是 2 秒；冻结的 1 秒窗口内 pose-valid adjacent pair 均为 0。它只证明当前 cohort 无法评价
  该机制，不证明时序或主动观测无效。
- Bonn RGB-D outcome-blind source audit 已修正到官方 marker/ROS/camera 位姿链，并在 26 个 parents 中找到
  25 个具备合法 pair 的 parents；从全序列均匀选出的 100 个 reference identities 不读取图像 payload，
  因此 source 的同预算一帧观测能力为 `POSE_PAIR_CAPABILITY_PASS`。
- positive-oracle R1 实际评价 56 references、504 queries，44 references 因几何不可观测 abstain；source-derived
  truth 为 `404 OCCUPIED / 2 CLEAR / 98 UNKNOWN`。static 只留下 2 个可恢复 positive opportunities，且它们
  与 2 个 CLEAR queries 都各自只覆盖 1 个 parent，未达到冻结的 4-parent/4-parent 分母门。
- 因此 passive/micro/task oracle 各自表面恢复 `2/2` 不得解释成有效增益；passive 与 micro 同时把 `2/2`
  CLEAR queries 错报为 OCCUPIED。所有臂间 decision 均保持 `null`，终态是
  `NOT_EVALUABLE_DATA_OBSERVABILITY_DENOMINATOR`；不得训练 learned scorer，也不得继续在 Bonn 上调门回救。
- R12 按同一冻结 `48/4/4` 门继续审计 TartanGround、ARKitScenes 与 TUM：前者 15 parents 仅 2 个满足
  micro pair；ARKit 21 个 pose-capable parents 却为 `219 OCCUPIED / 0 CLEAR / 105 UNKNOWN`；TUM native
  `640x480` 在 106 references 上仍为 `910 OCCUPIED / 0 CLEAR / 44 UNKNOWN`。跨分辨率复现说明旧标签要求
  `>=16` pixels 才 OCCUPIED、却要求 obstacle pixels 严格为 0 才 CLEAR，真实深度上的 1–15 pixel band
  结构性落入 UNKNOWN；三个 R12 terminal 均保留，不调门回救。
- R13 另立可证伪任务：同一 pose-only proposal pool 与一帧预算下，比较九个 body/path capsule 内新增的
  observed evidence cells，未观察 cell 保持 UNKNOWN。48 evaluable references 上，task oracle parent-macro
  `17.9569` cells/reference，高于 generic `14.0222` 与 passive `13.8847`；12 opportunity parents、10 strict-win
  parents、零 retention failure，终态 `TASK_CONDITIONED_QUERY_EVIDENCE_ORACLE_HEADROOM_PASS`。这首次证明
  task × next-pose 条件交互有可学上限，但还不是 learned policy。

## 当前证据入口

- [R10 terminal](TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_RESULT_2026-08-12.md)
- [R11 Phase-A independent validation](TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATION_RESULT_2026-08-13.json)
- [Top-24 result](TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_RESULT_2026-08-13.json)
- [FARO Phase-B implementation lock](TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_2026-08-13.md)
- [FARO Phase-B execution lock](TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK_2026-08-13.json)
- [FARO Phase-B formal result](TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_RESULT_2026-08-13.json)
- [Clear-observability single-axis Development result](TARO_CLEAR_OBSERVABILITY_SINGLE_AXIS_DEVELOPMENT_RESULT_2026-08-13.json)
- [Pair-support audit](TARO_TASK_DIRECTED_OBSERVABILITY_PAIR_SUPPORT_AUDIT_RESULT_2026-08-13.json)
- [Task-directed positive-oracle R1 result](TARO_TASK_DIRECTED_OBSERVABILITY_POSITIVE_ORACLE_CANARY_RESULT_2026-08-13.json)
- [Balanced-source frontdoors and R13 task-evidence oracle](TARO_TASK_OBSERVABILITY_BALANCED_SOURCE_FRONTDOOR_AND_QUERY_EVIDENCE_ORACLE_RESULT_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [TARO Module](../../../scripts/research/taro/README.md)

## 唯一 successor

`TARO_TASK_EVIDENCE_POSE_SCORER_R0`：

1. 冻结 scorer 输入只含 reference static evidence grid、候选相对 pose、相机内参和 source-time geometry；
   candidate neighbor depth 只能生成 FIT target 或在选择完成后评价，不能进入 EVALUATION selection；
2. 沿用 R13 的 14-parent namespace、pre-existing FIT/EVALUATION role 和每 reference 相同 pose-only proposal
   pool；所有非静态 arm 仍严格一帧，UNKNOWN 仍不是 negative；
3. 先只在 FIT parents 内选模型/正则，再在已消费 EVALUATION role 做明确标注的 Development replay；只有
   scorer parent-macro 同时高于 passive 与 generic、至少 4 parents 有 strict win、retention failure=0，才冻结
   candidate；
4. candidate 仍须在至少 4 个未被 R13 打开的 parents 上 confirmation 才能讨论 Android integration；不得把
   oracle target、EVALUATION depth 或 R13 全父源结果泄漏进 scorer input/training。

R11 outcome 只能作为已消费 Development evidence 做后验机制诊断；它不能改写上述 outcome-blind source
选择，也不能把 R11 改成 PASS。任何新的 dual-class confirmation 仍需 untouched parents。

## 当前允许

- 实现 `TARO_TASK_EVIDENCE_POSE_SCORER_R0`，scorer input 只含 reference static evidence、relative pose、内参与 source-time geometry；
- FIT parents 内训练、选模和交叉验证；在 EVALUATION role 上必须先选帧、后开 neighbor depth 评价；
- 为 scorer candidate 冻结至少 4 个 R13 未打开 parents 的 confirmation source lock；
- 对 consumed R11 evidence 做明确标注的只读后验机制诊断；
- 重放 hash-bound tests、validator 和只读 evidence 复核。

## 当前禁止

- 在 R7/R10 的 1 秒合法 pair 为 0 后训练时序模型或事后放宽窗口；
- 用不同额外帧预算比较 sensing arms，或只报告 recovery 而隐藏 false-occupied/known retention/cost；
- 在 Development canary 中输出 `CLEAR`、把 UNKNOWN 当 negative，或用 R11 outcome 选择该 canary 的 source；
- 回调 R12 的 query/label/gate，或把 R13 oracle/neighbor depth/EVALUATION target 泄漏进 scorer input/FIT；
- 将 R13 oracle PASS 冒充 learned scorer、untouched confirmation、Android 或产品成功；
- 修改 sealed R11 selection/selector/candidate/threshold，或覆盖、resume、删除、重跑已消费 one-shot；
- 越级训练、Android/QNN/HTP、默认 App、产品或安全结论。

## Claim ceiling

R13 已证明在 consumed TUM Development 上，task-conditioned next-pose oracle 比 passive/generic 增加更多 body/path
证据格；这是 task × pose 条件交互的正向机制证据。尚未证明可由 source-time scorer 预测，也未做 untouched
confirmation、移动端实现、产品有效性或用户安全验证；默认 App 不变。
