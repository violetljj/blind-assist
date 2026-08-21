# BlindAssist TARO

状态：`current / R31_V6_LEAK_INVALIDATED / R36_FRESH_PARENT_FAIL / R31_R35_FRESH_PARENT_REJECTED / R37_OPENLORIS_DEVELOPMENT_FAIL / R37_CLOSED / R38_REFERENCE_PREFLIGHT_PASS / R38_R32_FRESH_PARENT_CONFIRMATION_FAIL / TARO_CLOSED_NO_RESCUE / NO_ACTIVE_SUCCESSOR / CORE_SELECTOR_DEFAULT_OFF / DEFAULT_APP_UNCHANGED`

本页只维护 TARO 当前状态、权限和唯一算法 successor。较早完整 R0–R11 叙事保存在 [14d8ad7e 历史快照](archive/README_FULL_HISTORY_2026-08-13.md)，不能从中恢复旧权限。

## 当前主张

TARO 是独立并行 `WILD_LAB`：在声明的米制锚和冻结 factor/reducer 下，用低维 residual gauge posterior、可观测子空间和同预算额外观测，让 body/path-specific query 先于完整场景达到局部可识别。`UNKNOWN`、缺字段和不可观测方向永不转成 negative。

当前 Development 只检验 positive-occupancy task-directed observability，不输出 `CLEAR`，也不要求用户迈步取证。TARO 与 [Assistive Geometry](../assistive-geometry/README.md) 并列，不从 DepthART、Android、HTP 或默认 App 自动继承权限。

## 当前结论

- R10 以 dual-class coverage `NOT_EVALUABLE` 收口；不得改 selector、threshold、denominator 或 gate 回救。
- R11 exact-48 source-first Phase A、source-only 48→24 selection 与各自独立验证均 PASS；selected 24 identities 已不可变封存。
- selected-only FARO Phase B 已正式消费并通过独立复算：674/674 frames、6,066 queries、678-file root，unselected FARO=0。正式终态为 `NOT_EVALUABLE_DUAL_CLASS_COVERAGE`：28 个 definite-CLEAR queries 覆盖 7 parents，却只来自 10 physical frames，低于冻结的 12；10/10 clear-frame specificity 的单侧 95% Wilson 下界为 `0.787058`，低于 `0.8`。
- R11 与 R7 在所有 definite labels 上完全相同，只额外把 1 个 truth-UNKNOWN query 从 `OCCUPIED` 变为 `UNKNOWN`；weak-distal abstention 在 fresh cohort 没有产生预期的 clear-negative-control 效果。
- 单变量 post-hoc Development replay 只把 R9 `far_fraction_index` 从 0 改为 2：clear-frame recall 从 `0.60` 升至 `1.00`，eligible frames 从 34 增至 64、未超过冻结的 2× 上限 68；precision 从 `17.65%` 降至 `15.63%`。该候选保留为高召回 ranking proxy，不是 CLEAR classifier 或 confirmation。
- source-only pair-support audit 显示：R7 的 170 帧与 R10 的 710 帧虽然 pose 完整，但最小相邻间隔都是 2 秒；冻结的 1 秒窗口内 pose-valid adjacent pair 均为 0。它只证明当前 cohort 无法评价该机制，不证明时序或主动观测无效。
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
- R21 三折 macro 都高于 passive/generic，但 Bonn/ARKit 机会覆盖仅 `3/21`、`3/9`；R22 扩张 tensor 后
  held Bonn/ARKit 回归，learned scorer 保持 FAIL。
- R23 前瞻提交 109,426 参数 cross-attention、utility distribution 与 no-regret gate；ARKit/Bonn/TUM gated
  macro 为 `18.9294/19.2914/14.0333`，均高于 generic，但机会覆盖 `0/9`、`7/21`、`5/12`，三折仍 FAIL。
- R24 strict-opportunity teacher 在 held source 失校准；R25 解码 1,932 个 source-time RGB identities、增至
  258,450 参数后仅把 ARKit coverage 推到 `1/9`，Bonn/TUM macro 又回归。禁止在同一 1,690-candidate/
  57-parent 已消费表上调容量、loss、gate 或 handcrafted RGB 回救；详细数字见跟踪结果。
- R26 解析 disocclusion scorer 只在 ARKit 接近门槛；R27 将 reference RGB-D z-buffer 重投影到 candidate view，
  只在 body/path query 内计 warp hole/稳健光度不一致。它在 ARKit/TUM 六项门全过：macro
  `19.7961>16.4490`、`13.4896>13.3292`，breadth `5/5、8/6`；Bonn breadth `15/11` 却 macro
  `17.3000<19.2037`。R28 回归；R29 三源 macro 全胜但 breadth `4/5、0/11、5/6`；R30 仍失败。同表回调停止。
- R27 在独立 OpenLORIS `home1-1..5` 有效评测 25 references/5 parents/124 candidates，seal 前 candidate-depth reads=0；macro `16.32<17.36 generic`，4 个 opportunity parents 仅 1 个 strict win、门槛为 3。
  终态 `STOP_TARO_R27_OPENLORIS_HOME_FRESH_SOURCE_R2_FAIL`；R27 被 fresh-source 拒绝，OpenLORIS home 从此只作 consumed Development。
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
- outcome-blind backend preflight 在任何新 live model run 前锁定：冻结哈希的 YOLO11n/COCO 是唯一合格
  backend，因为其 `DetectorFrameResult` 能保留 source identity 并只输出 positive detections；当前相对深度
  backend 因无 source identity、逐帧归一化且无跨帧米制语义被拒绝。正式 shadow 要求 4 个 opaque scene
  parents、至少 120 个 evaluable references；current+passive 与 current+pose-diverse 各自严格只多用一帧，
  primary 是中心下方 screen-space proxy 内新增的正对象 token。无检测保持空 positive set，不是 negative/safe。
- 修正 preliminary `scene_a..d` 中把 focus 状态误纳入 token identity 的实现偏差后，使用 fresh
  `scene_e..h` 完成正式 4-scene/120-reference device shadow：所有 denominator/runtime gate 通过；pose-diverse
  parent macro 新增 focused token 为 `0.5750`，passive 为 `0.3917`，pose 在 4 个 scene parents 中 3 胜 1 负，
  terminal 为 `POSE_DIVERSE_POSITIVE_VISUAL_EVIDENCE_PASS`。这只证明单设备受控场景中冻结模型的屏幕空间
  positive-evidence observability，不证明 detector accuracy、body/path 几何、碰撞正确性、产品或安全。
- 默认 App source audit 显示 CameraX 接缝不承载同帧 ARCore pose；SharedCamera protocol 虽已锁定，但在当前算法优先级下不再是 active successor，默认 App 不变。
- R38 reference-input-only preflight 在 candidate RGB/model、candidate sensor depth 与 task outcome 均未打开时，
  冻结出全局 role-disjoint 的 `11 parents / 25 references / 112 unique candidates`，通过预冻的
  `8-parent / 24-reference` 分母门。唯一一次 R32 confirmation 随后有效消费：ranker parent-macro
  `22.0682 < 25.0227 generic`，虽高于 passive `15.3561`，但 5 个 opportunity parents 仅 2 个 strict win，
  同时失败 macro 与 `max(3, 50%)` strict-win 门。其余分母、同预算、retention、identity 和 score-before-target
  firewall 全部通过；终态是有效算法确认 FAIL，不是实验失效。R32 不再允许用 R38 回调或 rescue，TARO 当前关闭且无 active successor。

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
- [R23-R25 complex query-conditioned scorer results](TARO_COMPLEX_QUERY_CONDITIONED_SCORER_RESULTS_2026-08-14.json)
- [R26-R30 reprojection results](TARO_REPROJECTION_VISIBILITY_SCORER_RESULTS_2026-08-14.json) · [OpenLORIS home fresh-source result](TARO_R27_OPENLORIS_HOME_FRESH_SOURCE_RESULT_2026-08-14.json)
- [R31-v7 / R35 zero-leakage Development and R36 fresh-parent rejection](TARO_R31_R36_RELIABILITY_REGIME_AND_FRESH_PARENT_RESULT_2026-08-15.json)
- [R37 pose-constrained geometry rejection and R38 fresh ARKit source checkpoint](TARO_R37_R38_POSE_GEOMETRY_AND_FRESH_ARKIT_SOURCE_RESULT_2026-08-15.json)
- [R38 R32 fresh-parent confirmation failure](TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_RESULT_2026-08-15.json) · [execution lock](TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_EXECUTION_LOCK_2026-08-15.json)
- [Pose-diverse portfolio and default-off core selector](TARO_POSE_DIVERSE_BASELINE_PORTFOLIO_AND_CORE_SELECTOR_RESULT_2026-08-13.json)
- [Historical isolated canary preflight and superseded device environment stop](TARO_POSE_DIVERSE_SELECTOR_ISOLATED_CANARY_PREFLIGHT_RESULT_2026-08-13.json)
- [ARCore device selector, raw-depth stop and RGB pair-support result](TARO_POSE_DIVERSE_ARCORE_DEVICE_AND_RGB_PAIR_RESULT_2026-08-13.json)
- [Owned RGB history exact-identity and cost result](TARO_RGB_FRAME_HISTORY_RETENTION_AND_COST_RESULT_2026-08-14.json)
- [Exact selected/reference delayed-decode integrity result](TARO_RGB_SELECTED_PAYLOAD_DECODE_INTEGRITY_RESULT_2026-08-14.json)
- [Frozen YOLO positive-evidence backend preflight and shadow lock](TARO_RGB_PAIR_FROZEN_VISUAL_EVIDENCE_BACKEND_PREFLIGHT_2026-08-14.json)
- [Frozen YOLO multi-scene positive-evidence shadow result](TARO_RGB_PAIR_YOLO_POSITIVE_EVIDENCE_SHADOW_RESULT_2026-08-14.json)
- [Default-off App source architecture audit and SharedCamera canary lock](TARO_DEFAULT_OFF_APP_POSITIVE_EVIDENCE_SHADOW_PREFLIGHT_RESULT_2026-08-14.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [TARO Module](../../../scripts/research/taro/README.md)

## 唯一 successor

无。`TARO_R38_R32_ARKITSCENES_FRESH_PARENT_CONFIRMATION_R0` 已消费并关闭：

1. R31-v6 的 inference opportunity anchor 读取了 target-derived coverage，相关零泄漏主张已作废；R31-v7 修复后，R35 source-regime policy 只在 consumed Development 四源通过；
2. R36 在未见的 OpenLORIS corridor1-3..5 上零泄漏执行有效，但 ranker 与 generic 同为 `31.8333`，strict-win parents `0/3`，opportunity parents `2/3`，终态 fresh-parent FAIL；R31/R35 不再允许在这些 parent 上回调；
3. R37 的 pose-constrained candidate scale+shift 在 consumed OpenLORIS home 上只得 `16.16 < 17.36` generic，strict-win parents `2/5 < 3/5`；预先有界的八种高容量 LOPO 组合也均未超过 generic，因此 R37 已关闭；
4. R32 的 reference-sensor-anchored frozen metric monocular geometry 在 consumed ARKitScenes 上曾取得 `21.9392 > 16.4490` 且 strict-win parents `5/17`；
5. R38 在 12 个全新 Validation parents 上先完成 reference-only preflight，再冻结 `11/25/112` role-disjoint identity。唯一一次 confirmation 得到 `22.0682 < 25.0227 generic`、strict-win `2/5`，有效 FAIL；不得用 R38 outcome 改模型、阈值、候选或 gate，也不建立 R39/R40/R41 rescue。

R11 outcome 只能作为 consumed Development 做后验诊断，不能改写 source selection 或 R11 terminal；任何 dual-class confirmation 仍需 untouched parents。

## 当前允许

- 保留 R27、R31/R35、R36 和 R37 的 exact identities、selection seal、基线、分母和失败终态，禁止覆盖重跑；
- 只读复核 R38 preflight、execution lock、selection seal、result 与 hash binding；
- 将 R31–R38 负结果链和 DepthART D3R6 可信 Development candidate 用于论文主线盘点，但不自动恢复任何算法执行权限；
- 对 consumed R11 evidence 做明确标注的只读后验机制诊断；
- 重放 hash-bound tests、validator 和只读 evidence 复核。

## 当前禁止

- 在 R7/R10 的 1 秒合法 pair 为 0 后训练时序模型或事后放宽窗口；
- 用不同额外帧预算比较 sensing arms，或只报告 recovery 而隐藏 false-occupied/known retention/cost；
- 在 Development canary 中输出 `CLEAR`、把 UNKNOWN 当 negative，或用 R11 outcome 选择该 canary 的 source；
- 回调 R12/R19/R20/R21 的 query/outcome/gate，或把 neighbor depth 泄漏进 scorer input；
- 在 R23–R30 已消费表上调容量、loss、seed、gate、background correction 或 RGB channels，并把 post-hoc fold 改善包装成 confirmation；
- 在 OpenLORIS home1-1..5 或 corridor1-3..5 上回调 R31/R35/R36，或用 R36 outcome 选择 R37 参数；
- 将 candidate sensor depth、target coverage、target gain 或任何 outcome-derived anchor 输入 R37 的 scale、shift、score、gate 或 selection；
- 在 R37 上继续调模型容量、特征组合或 LOPO estimator；覆盖或重跑 R38 one-shot，或用 R38 结果回调 R32；
- 建立 R39/R40/R41 或其他 TARO rescue continuation；新 TARO 问题必须由用户重新授权 materially different hypothesis 和新协议；
- 将 generic core selector 或本次单设备 screen-space PASS 写成 task-specific scorer、跨设备成功、任务增益、
  风险融合、默认 App 或产品成功；
- 把独立 ARCore Session 直接并挂到当前 CameraX analyzer，或在缺少 exact ARCore pose 时运行已验证 selector；
- 将重投影/旧 raw depth 当成独立 fresh observation，或在当前设备上继续回调 raw-depth pair gate；
- 修改 sealed R11 selection/selector/candidate/threshold，或覆盖、resume、删除、重跑已消费 one-shot；
- 越级训练、Android/QNN/HTP、默认 App、产品或安全结论。

## Claim ceiling

R13 有 task-conditioned oracle headroom；R27 已被 fresh-source 拒绝。R31-v7/R35 被 R36 fresh-parent 拒绝；R37 在 consumed OpenLORIS 上回归并已关闭。R38 又以有效的 parent-disjoint ARKitScenes Validation confirmation 拒绝冻结 R32：`22.0682 < 25.0227 generic`、strict-win `2/5`。当前 TARO 没有算法突破、没有 active successor，也不能以 rescue tuning 续命。该负结果不证明 task-directed observability 整体不可能；设备证据仍不替代 fresh RGB-D+pose 验证，未证明 detector accuracy、碰撞、产品或安全，默认 App 不变。
