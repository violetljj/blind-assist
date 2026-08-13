# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / R13_TASK_EVIDENCE_ORACLE_HEADROOM_PASS / R14_R20_SCORER_TRANSFER_NOT_CONFIRMED / CROSS_SOURCE_LEARNED_RANKER_ACTIVE / DEFAULT_APP_UNCHANGED`

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
- R14 pointwise ridge、R15 pairwise ridge 和 R16 fixed analytic scorer 都未越过冻结的跨父级/跨源基线门；
  R15 FIT gate 失败时 Bonn target reads 保持 0，负终态均保留。
- R18 在全部已消费 TUM/Bonn Development 上从 96 个预冻混合候选中只找到一个跨源 admissible policy：
  `translation_unit + 0.8*visible_unknown_unit + 0.05*rotation_unit`。但其 Bonn 相对 generic 优势只有 `+0.0191`，
  因此只授权新任务结果确认，不授权 Android。
- R19 四个 task-outcome-blind TUM parents 上，冻结 policy 的 parent-macro `11.6375` 同时高于 generic
  `11.25` 与 passive `8.9375`，但 strict-win parents 只有 2、低于预冻 3；正式终态仍为 FAIL。
- R20 在尚未打开 task-evidence neighbor outcomes 的 ARKitScenes 上按机会分母重做确认：40 references、17 parents、
  9 opportunity parents，policy 只覆盖 2，且 macro `16.3363` 低于 generic `16.4490`；oracle 仍为 `25.0863`。
  这把问题定位为 scorer transfer，而不是任务无 headroom；Android 与默认 App 仍未授权。

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
- [R14-R20 scorer and confirmation results](TARO_TASK_EVIDENCE_SCORER_AND_CONFIRMATION_RESULTS_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [TARO Module](../../../scripts/research/taro/README.md)

## 唯一 successor

`TARO_CROSS_SOURCE_LEARNED_RANKER_R0`：

1. 训练数据只可来自现已消费的 TUM、Bonn、ARKitScenes task-evidence outcomes；scorer 输入仍只含 reference
   static evidence、relative pose、内参与 source-time geometry，neighbor depth 只作 target；
2. 模型族与超参数必须先冻结，再做 leave-one-source-family-out；每个 held source 都必须在 parent-macro 上同时
   高于 passive 与 generic，并按 opportunity denominator 覆盖足够 strict-win parents；
3. 不通过三源逐一外推就停止，不得在 held source 上调模型；通过后才锁一个未打开 task-evidence outcome 的
   新来源/父级组；
4. fresh confirmation 全门通过前，禁止 Android integration、默认 App 或产品主张。

R11 outcome 只能作为已消费 Development evidence 做后验机制诊断；它不能改写上述 outcome-blind source
选择，也不能把 R11 改成 PASS。任何新的 dual-class confirmation 仍需 untouched parents。

## 当前允许

- 实现 `TARO_CROSS_SOURCE_LEARNED_RANKER_R0`，输入只含 reference static evidence、relative pose、内参与 source-time geometry；
- 在已消费 TUM/Bonn/ARKitScenes 上做 leave-one-source-family-out，held source 不参与拟合或调参；
- 三源门均过后，为 frozen learned candidate 锁新 task-outcome-blind confirmation source；
- 对 consumed R11 evidence 做明确标注的只读后验机制诊断；
- 重放 hash-bound tests、validator 和只读 evidence 复核。

## 当前禁止

- 在 R7/R10 的 1 秒合法 pair 为 0 后训练时序模型或事后放宽窗口；
- 用不同额外帧预算比较 sensing arms，或只报告 recovery 而隐藏 false-occupied/known retention/cost；
- 在 Development canary 中输出 `CLEAR`、把 UNKNOWN 当 negative，或用 R11 outcome 选择该 canary 的 source；
- 回调 R12/R19/R20 的 query、outcome 或 gate，或把 neighbor depth 泄漏进 scorer input；
- 将 R13 oracle、R18 Development PASS 或两次 confirmation FAIL 冒充 learned scorer、Android 或产品成功；
- 修改 sealed R11 selection/selector/candidate/threshold，或覆盖、resume、删除、重跑已消费 one-shot；
- 越级训练、Android/QNN/HTP、默认 App、产品或安全结论。

## Claim ceiling

R13 已证明 task-conditioned next-pose oracle headroom；R19 在 TUM 的宏平均正迁移未越过 strict-parent 门，
R20 在 ARKitScenes 又未越过 generic 与 opportunity 门，因此 source-time scorer 的可迁移性尚未确认。
当前只授权跨已消费三源的 learned-ranker Development；未授权移动端实现、产品有效性或用户安全验证，默认 App 不变。
