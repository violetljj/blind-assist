# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / R13_ORACLE_HEADROOM_PASS / R14_R22_TASK_SCORER_TRANSFER_FAIL_STOP / POSE_DIVERSE_BASELINE_MULTI_SOURCE_PASS / ANCHOR_DEVICE_CANARY_PASS / RGB_PAIR_SUPPORT_PASS / RGB_HISTORY_RETENTION_COST_PASS / RGB_SELECTED_DECODE_INTEGRITY_PASS / FRESH_RAW_DEPTH_PAIR_FAIL_STOP / CORE_SELECTOR_DEFAULT_OFF / DEFAULT_APP_UNCHANGED`

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
- R21 固定 3-seed bounded nonlinear ranker 做 leave-one-source-family-out：三折 parent-macro 都同时高于
  passive/generic，但 Bonn 只覆盖 `3/21` opportunity parents、ARKit 只覆盖 `3/9`，低于冻结的半数覆盖门，
  因此 learned scorer 仍 FAIL。R22 只增加 query/along/height tensor、保持网络和门不变，held Bonn/ARKit
  反而回归，证明当前样本下高维表示过拟合；没有继续扩模型容量。
- 与 task scorer 分开，预先定义的 pose-only generic arm 在四个 cohort 均高于 passive：旧 TUM
  `14.0222>13.8847`、Bonn `19.2037>17.2662`、task-outcome-blind TUM `11.25>8.9375`、ARKit
  `16.4490>12.9431`。因此 `TARO_POSE_DIVERSE_GENERIC_R0` 已实现为 `core:ustrf` 中默认关闭的纯 Kotlin
  frame selector；它只返回历史 frame identity，不读 payload、不融合风险、不发提醒、不接默认 App。
- 隔离 benchmark 现有两条互不越权的准入：既有 `UstrfVioPoseAdmission` 继续保留外参门禁，供未来风险场
  链路使用；新增的 `TaroArCoreAnchorPoseAdmission` 只把同一 ARCore session、同一正在跟踪的 local Anchor
  下的相机相对位姿交给纯 camera-history selector。后一条不做 body-frame warp，所以不伪造或要求外参，
  也不能反向授权风险融合。时间戳不前进、连续跟踪 warm-up 不足、Anchor 非 TRACKING、相对位姿退化或
  任一 admission failure 都不会进入历史 buffer。
- 项目自有 `TaroArCoreAnchorPoseDiverseCanaryTest` 已在 `SM-S9280 / Android 16 / ARCore 1.54.260890093`
  真机通过：600 attempts 中 547 个 anchor-pose admissions、543 次合法选择，选择窗
  `199.98ms..999.97ms`，最大位移 `0.1127m`、最大偏航 `0.1514rad`；这关闭了先前 device ENV_BLOCKED。
- fresh raw-depth payload 路线在 `RAW_DEPTH_ONLY` 下正式 FAIL：474 个 candidate 只有 1 个严格同 source-frame
  的新深度，且发生在 pose warm-up 前，fresh pose-bound frames/pairs 都为 0。不得把 473 个重投影/旧深度
  当作独立额外观测回救。
- RGB payload pair 路线在同一真机通过：595 张 camera images 中 564 张与已准入 Anchor pose 共享同一个
  current ARCore Frame API provenance，564 个 bounded luminance digests 全部不同，形成 560 次合法选择；
  时间窗 `166.03ms..999.99ms`，最大位移 `0.1457m`、最大偏航 `0.2379rad`。Image/Frame/Camera2 三种
  timestamp 关系作为诊断保留，不用 nearest-frame 绑定；该轮尚未保留或解码任何 RGB frame。
- 后继 retention/cost canary 已把 465 个 `640x480 YUV_420_888` payload 在 `Image.close()` 前完整复制到
  benchmark 自有历史，461/461 次选择都以完整 `UstrfFrameStamp` 精确反查 payload receipt，identity miss、
  resource error 均为 0。1 秒/32 MiB 双限下峰值为 28 帧、`17,203,144` bytes，438 次均按 source-age
  正常淘汰、byte-cap 淘汰为 0；copy+append+select p50/p95 为 `2.602/3.810ms`。这证明 payload ownership
  与单机成本可行，尚未解码像素、运行模型或证明任务增益。
- delayed-decode canary 随后对 556/556 个 exact selected/reference pairs 运行既有冻结
  `D45Yuv420ToRgbaDecoder`：所有 source identity、尺寸和 selected 重放 RGBA hash 都一致，decode/resource
  error 为 0；556 个 reference RGBA hash 全部不同，selected 覆盖 201 个不同 hash。单次 CPU decode
  p50/p95 为 `18.045/30.358ms`，瞬态 RGBA 为 `1,228,800` bytes；三解码完整性探针 p50/p95
  `57.387/67.152ms`，不是建议的产品 cadence。尚未运行任何 detector/depth model。

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
- [R21-R22 cross-source learned-ranker result](TARO_CROSS_SOURCE_LEARNED_RANKER_RESULT_2026-08-13.json)
- [Pose-diverse portfolio and default-off core selector](TARO_POSE_DIVERSE_BASELINE_PORTFOLIO_AND_CORE_SELECTOR_RESULT_2026-08-13.json)
- [Historical isolated canary preflight and superseded device environment stop](TARO_POSE_DIVERSE_SELECTOR_ISOLATED_CANARY_PREFLIGHT_RESULT_2026-08-13.json)
- [ARCore device selector, raw-depth stop and RGB pair-support result](TARO_POSE_DIVERSE_ARCORE_DEVICE_AND_RGB_PAIR_RESULT_2026-08-13.json)
- [Owned RGB history exact-identity and cost result](TARO_RGB_FRAME_HISTORY_RETENTION_AND_COST_RESULT_2026-08-14.json)
- [Exact selected/reference delayed-decode integrity result](TARO_RGB_SELECTED_PAYLOAD_DECODE_INTEGRITY_RESULT_2026-08-14.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [TARO Module](../../../scripts/research/taro/README.md)

## 唯一 successor

`TARO_RGB_PAIR_FROZEN_VISUAL_EVIDENCE_BACKEND_PREFLIGHT_R0`：

1. 在读取新的 live model output 前，只读审计现有冻结 RGB backends；最多选择一个能保留完整 source-frame
   identity、输出 positive-only visual evidence 且语义上与 body/path observability 有明确交集的 backend；
2. 预先冻结 current-only、current+passive 与 current+pose-diverse 三臂的同一额外帧预算、机会/abstention
   分母、延迟/内存 receipt 和停止条件；UNKNOWN 不得转成 negative；
3. 若没有合格 backend，必须以 `NOT_EVALUABLE_NO_ADMISSIBLE_FROZEN_BACKEND` 停止；不得换模型、训练、用
   live output 事后定义指标，或把 pipeline readiness 写成任务增益；
4. learned task scorer 保持 STOP，只有 materially new source-time signal/supervision 才可重开；不得用 generic
   baseline 的落地掩盖 task-specific scorer 失败。

R11 outcome 只能作为已消费 Development evidence 做后验机制诊断；它不能改写上述 outcome-blind source
选择，也不能把 R11 改成 PASS。任何新的 dual-class confirmation 仍需 untouched parents。

## 当前允许

- 对现有冻结 RGB backends 做 outcome-blind 只读 preflight，并预注册唯一 visual-evidence shadow protocol；
- 对纯 camera-history canary 使用同 session/same-anchor 相对位姿；外参门禁继续用于需要 body-frame/risk-field
  warp 的独立链路，不得把两者混为同一权限；
- 只有 materially new source-time signal/supervision 才可另立 learned scorer successor；
- 对 consumed R11 evidence 做明确标注的只读后验机制诊断；
- 重放 hash-bound tests、validator 和只读 evidence 复核。

## 当前禁止

- 在 R7/R10 的 1 秒合法 pair 为 0 后训练时序模型或事后放宽窗口；
- 用不同额外帧预算比较 sensing arms，或只报告 recovery 而隐藏 false-occupied/known retention/cost；
- 在 Development canary 中输出 `CLEAR`、把 UNKNOWN 当 negative，或用 R11 outcome 选择该 canary 的 source；
- 回调 R12/R19/R20/R21 的 query、outcome 或 gate，或把 neighbor depth 泄漏进 scorer input；
- 将 generic core selector 写成 task-specific scorer、跨设备成功、任务增益、风险融合、默认 App 或产品成功；
- 将重投影/旧 raw depth 当成独立 fresh observation，或在当前设备上继续回调 raw-depth pair gate；
- 修改 sealed R11 selection/selector/candidate/threshold，或覆盖、resume、删除、重跑已消费 one-shot；
- 越级训练、Android/QNN/HTP、默认 App、产品或安全结论。

## Claim ceiling

R13 已证明 task-conditioned oracle headroom；R21 证明 learned scorer 可跨源提高宏平均，但没有广泛覆盖机会父级，
R22 表示扩张又回归，因此 task-specific scorer 停止。pose-diverse generic baseline 已获得跨三源族的 Development
支持并落为默认关闭的纯 Kotlin selector；单台真实 ARCore 设备已证明 Anchor 相对选帧与 source-bound RGB pair
support，fresh raw-depth pair 则失败；同机已进一步证明完整 YUV payload 的 1 秒/32 MiB 有界 ownership、
selection identity、copy 成本和 exact selected/reference 延迟解码完整性。尚未运行模型、证明任务证据增益、
完成风险融合、产品有效性或安全验证，默认 App 不变。
