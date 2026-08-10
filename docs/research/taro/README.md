# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / P0_PASS / O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS / O0R_SOURCE_TRUTH_MATERIALIZED / O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE / DEPTHART_CANDIDATES_239_SEALED / PARTIAL_FACTOR_CANARY_COMPLETE / APPLE_SCALE_SOURCE_CANARY_COMPLETE / SOURCE_ANCHORED_FACTOR_CANARY_R1_COMPLETE / UNCONDITIONAL_PRE_SCALE_NOT_ADOPTED / SOURCE_SCALE_239_OF_239 / ORACLE_PAIRED_166 / NO_ACTIVE_EXECUTION / DEFAULT_APP_UNCHANGED`

## 需求、使用场景与效果合同

TARO 不试图定义全部视障出行需求。它只选择一个与当前算法对象直接对应的窄场景：用户已经通过
白杖、定向行走技能或外部导航掌握宏观方向，需要判断前方几步内，自己的身体沿声明路径是否具有
足够净空。它不承担全局导航、道路穿越、自动接管行走或“安全路线”认证。

| 需求 | 系统应形成的效果 | 研究主指标 | 对算法的直接约束 |
|---|---|---|---|
| 不把走不通说成能走 | 路径/身体特定的完整净空区间 | `false-clear`、query error | 不能只用均值越阈值 |
| 不把能走的地方普遍阻断 | 在不增加错误放行时减少保守阻断 | `false-block`、known coverage | 不能靠 all-`UNKNOWN` 通过 |
| 证据不足时不装作知道 | 校准的 `UNKNOWN` 与 reason code | interval calibration、错误高置信率 | 只更新可观测状态方向 |
| 不要求用户冒险迈步取证 | 先复用被动历史，必要时才考虑站定相机微基线 | query-risk reduction、prompt/time cost | 禁止 body motion 取证 |

这张表是需求到研究效果的追踪合同，不是已经验证的用户效果。当前 O0M 只证明冻结 synthetic
analytic family 上的 mechanics；真实 query Pareto、交互收益和用户结果仍未建立。

## 当前主张

TARO（Task-directed Active Risk Observability，任务定向主动风险可观测性）研究：

> 在有效相机/裁剪/旋转 receipt 与至少一个独立米制锚存在时，即使完整场景与完整
> gauge 不能被唯一恢复，body-swept clearance 查询能否先达到可识别、可校准状态；
> 当查询仍不可识别时，受限的被动帧复用或站定相机微基线，能否比通用熵、最大视差
> 或普通 next-best-view 更有效地降低任务决策风险，并在证据不足时保持 `UNKNOWN`？

本版本以 `task-query identifiability` 为主科学命题，并用两个受共同协议约束的组件检验它：

- `GaugeFix`：metadata-first 的低维残余 gauge posterior、协方差与可观测子空间更新；
- `PARA`：只有被动 query 仍不可识别且 action oracle 先通过时，才研究以
  body/path-specific clearance query 为目标的受限观察与证据选择。

TARO 明确接受**组合式创新**：factor encoder、残余 gauge posterior、可观测子空间求解、
选择性校准和 action scorer 都可以继承或替换优秀前作。原创性不要求每个零件从零发明，而要求
它们服从同一条可证伪合同：`clearance functional → identifiability → calibrated uncertainty →
evidence value → deterministic reducer`。组合是否成立，必须用同预算强基线、可插拔替换、单组件/
联合 factorial ablation 和联合增益归因来证明，不能把各模块单独有效直接写成系统有效。

两个组件不得在 outcome 后任意拆分或互相背书；active branch 若失败，任何 passive-only 延续都必须
另立版本。这样可保持贡献主次清楚，同时不把主动提示预设为 TARO 必须成立的用户行为。

TARO 是与 [Assistive Geometry](../assistive-geometry/README.md) 并列的独立
`WILD_LAB` 算法路线，不是其 F1/F2 successor，也不修改其 frozen factor schema、
`FactorTensorAdapter` blocker、deterministic reducer、数据角色、终态或唯一 successor。

## 唯一真源与稳定入口

- 本页：路线状态、权限、唯一 successor、禁止动作与 claim ceiling；
- [TARO R0 详细路线指南](TARO_R0_RESEARCH_ROUTE_GUIDE_2026-08-10.md)：研究命题、
  数学定义、接口、阶段、数据、基线、指标、拟议 kill gates 与停止条件；
- [TARO P0 协议锁](TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK_2026-08-10.md)：
  四个 schema、measurement-only identifiability、有限 task ambiguity、八臂 factorial、负控、
  数据角色、gate 与权限；
- [TARO P0 lock result](TARO_P0_PROTOCOL_LOCK_RESULT_2026-08-10.md)：33 个静态/mutation tests
  通过；科学状态仍为 `NOT_RUN`；
- [TARO O0M protocol lock](TARO_O0M_SYNTHETIC_IDENTIFIABILITY_AND_FACTORIAL_MECHANICS_PROTOCOL_LOCK_2026-08-10.md)：
  冻结 10 个 identifiability cases、5 scenes × 8 arms × 2 modes、十门与 one-shot 预算；
- [TARO O0M protocol result](TARO_O0M_PROTOCOL_LOCK_RESULT_2026-08-10.md)：33/33 mutation tests；
  protocol lock 时 implementation、runner 与 scientific artifact 尚不存在；
- [TARO O0M implementation lock](TARO_O0M_IMPLEMENTATION_LOCK_2026-08-10.md)：独立 NumPy runtime
  与 13/13 disjoint unit tests 已 hash-bound；
- [TARO O0M one-shot execution lock](TARO_O0M_ONE_SHOT_EXECUTION_LOCK_2026-08-10.md)：exact
  code/fixture/argv/environment/resource/root 已绑定；该 one-shot 现已消费；
- [TARO O0M signed result](TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_RESULT_2026-08-10.md)：
  `92/92` records、`10/10` gates 与两次 byte-identical replay PASS；真实 O0R 仍不可评估；
- [TARO O0R ARKitScenes source-and-adapter contract](TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.md)：
  新的 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE` fresh TRAIN parent、FARO truth、uncertainty、
  query、factor injection、统计、预算与 failure scope；
- [TARO O0R source-and-adapter lock result](TARO_O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK_RESULT_2026-08-10.md)：
  pinned exclusion/roster 与关键 implementation seam 可重算，21/21 mutation tests；scientific status 仍为 `NOT_RUN`；
- [TARO O0R source-adapter implementation lock](TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK_2026-08-10.md)：
  纯内存 runtime 与 44/44 synthetic focused tests 已 hash-bound；source I/O 与 scientific execution 仍为 false；
- [TARO O0R truth-only preflight lock](TARO_O0R_ARKITSCENES_TRUTH_ONLY_ONE_SHOT_PREFLIGHT_LOCK_2026-08-10.md)：
  精确冻结 24-parent × 3-asset 的 72 个 HEAD target、静态 argv、环境、预算、授权缺口和四个 absent roots；
  `HEAD_NOT_RUN / ONE_SHOT_UNCONSUMED / EXECUTION_NOT_AUTHORIZED`；
- [TARO O0R data-use authorization](TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-10.json)：
  用户原文授权锁定 24 个 Training video 的 HEAD、bounded source/truth-only WILD_LAB 使用；授权本身不激活 runner；
- [TARO O0R materializer amendment](TARO_O0R_ARKITSCENES_MATERIALIZER_INPUT_AND_PERSISTENCE_AMENDMENT_LOCK_2026-08-10.md)：
  冻结 all-exact frame roster、fit-before-eval、per-query lookup、original-member envelope 与 ndarray reload gate；
- [TARO O0R truth materializer implementation lock](TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_IMPLEMENTATION_LOCK_2026-08-10.md)：
  HEAD/source/truth runner、atomic writer 与 25/25 synthetic focused tests 已 hash-bound；HEAD/source/truth 仍未运行；
- [TARO O0R Content-Length HEAD execution lock](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK_2026-08-10.md)：
  Attempt 01 的精确 72 URL、zero-body、argv/environment/budget 与 exclusive root；现已 pre-start superseded；
- [TARO O0R HEAD Attempt 01 pre-start incident](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_ATTEMPT_01_PRESTART_INCIDENT_2026-08-10.md)：
  `artifacts.local` junction 被旧 path guard 误判；发生在 root/HEAD 前，request=0、one-shot 未消费；
- [TARO O0R Content-Length HEAD execution lock Attempt 02](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_EXECUTION_LOCK_ATTEMPT_02_2026-08-10.md)：
  绑定 junction-aware implementation commit `2c0fdef8` 与同一 72-URL plan；当前 `AUTHORIZED_UNCONSUMED`；
- [TARO O0R Content-Length HEAD result Attempt 02](TARO_O0R_ARKITSCENES_CONTENT_LENGTH_HEAD_RESULT_ATTEMPT_02_2026-08-10.md)：
  71/72 available；`47333152/lowres_wide.traj` 3/3 HTTP 403；zero body、HEAD one-shot consumed、no replacement；
- [TARO O0R Apple scale source canary R0 result](TARO_O0R_ARKITSCENES_APPLE_SCALE_SOURCE_CANARY_R0_RESULT_2026-08-11.md)：
  239/239 source-only estimates、166 paired frames、16/16 parents improved；parent-macro absolute log-scale error
  从 `0.30498` 降到 `0.01561`，但仍仅为 retrospective WILD_LAB diagnostic；
- [TARO O0R source-anchored factor canary R1 result](TARO_O0R_ARKITSCENES_SOURCE_ANCHORED_FACTOR_CANARY_R1_RESULT_2026-08-11.md)：
  171 frames / 1,539 queries；尺度后 SUPPORT height 与 BOUNDARY error 显著改善，但新增 112 个
  extractor loss、恢复 0，故不采用 unconditional pre-scale；R1A 仅修正 canonical summary round-trip；
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)：项目级算法路线登记；
- [R2 factorized geometry protocol](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)：
  可只读复用的 factor/reducer/UNKNOWN 上游合同；
- [A0 failure anatomy](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)：
  只作为系统性 conservative-bias 诊断，不作为 TARO 选模或阈值来源；
- [AG-DCA result](../assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.md)：
  当前 target-atlas timestamp/pose/factor-support 缺口的能力真源。

## 唯一 successor

`APPLE_SEEDED_SUPPORT_RECOVERY_CANARY_R2`

O0M implementation、one-shot lock 与唯一正式执行均已完成。正式 `10+80+2` synthetic canary
终态为 `TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS`；exclusive root 已创建并消费，结果不得
覆盖、删除或重跑。

前一版本的真实 O0R 硬终态 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE` 已由新的、outcome 前
冻结的 ARKitScenes source-and-adapter contract 在**协议层**关闭：24 个全新且 visit-disjoint 的
TRAIN parent 已按 pinned exclusion + SHA 排序分配为 `8 ADAPTER_FIT + 16 O0R_EVAL_CANDIDATE`；
FARO-only factor/query truth、real-residual uncertainty、exact timestamp/K/pose receipt、八臂
deterministic injection、parent-level statistics、预算和 failure scope 均已冻结并通过静态验证。

纯内存 adapter 已以 44/44 synthetic focused tests 关闭其 implementation seam：覆盖 exact
timestamp/pose watermark、冻结 role/roster/asset/decoded-content binding、由 FARO support geometry 生成的
source+9-query receipt、由绑定 FARO/AppleDepth/confidence 内推的 fit-only uncertainty、FARO 与
candidate-depth 双 extractor、deep-read-only whole geometry/immutable base/common support、sparse
BOUNDARY、固定 model/checkpoint/metric-zero candidate output receipt、独立 TARO query reducer、genuine
truth-only 9/9 bundle 与带逐组件 lineage/parent context 的 8×2 injection；公开 API 不提供 caller residual、
factor reseal 或 self-signed truth-only scale correction。该 Module 本身不包含 downloader、archive reader、
materializer、DepthART runner、scientific evaluator 或 artifact writer；source I/O 由独立 materializer Module 承担。

truth-only preflight lock 现已静态通过：它从合同重算 24 个 parent，并以 3 个 URL template 展开
72 个唯一 Training HEAD target；request-plan SHA、离线复验 argv、Python/NumPy/SciPy/Pillow 环境、
20 GiB/50 GiB/12 h/12 GiB/2 GiB 上限和四个 absent roots 均已冻结。该锁没有发送 HEAD，未读取
Content-Length，也未创建或消费 one-shot root；绑定的旧 Assistive Geometry receipt 只覆盖 6 个旧视频且
缺 trajectory，不能授权 TARO body access。此外 `47333152` 被官方 downloader 列为缺少 3DOD assets，
未来 trajectory HEAD 若非 `200 + Content-Length`，R0 必须 `NOT_EVALUABLE`，不得换 parent。

用户随后以原文授权该锁定 roster 的 HEAD 与 bounded source/truth-only WILD_LAB 使用；outcome-blind
amendment 进一步冻结 all-exact denominator、8-parent fit-before-eval、每 query 独立 FARO/confidence lookup、
official member provenance 与 content-addressed ndarray reload。独立 materializer Module 已实现 HEAD/source/truth
runner、atomic one-shot writer，并通过 25/25 focused tests（含消费后故障与 junction containment）与 6/6 lock-validator mutation tests；未发送 HEAD/GET、
未打开 selected source、未创建 root、未拟合真实 uncertainty 或物化 truth。

HEAD Attempt 01 已提交，但在任何 HEAD 前由旧 `safe_join` 将仓库授权的 F-backed `artifacts.local`
junction 误判为 `PATH_ESCAPE`。失败发生在 output root 解析阶段，因此 `HEAD/GET/body/source/truth=0`、
五个正式 root 仍不存在、one-shot 未消费；Attempt 01 原锁不得原地重跑。实现现已仅对受信任
`artifacts.local` namespace 做 junction-aware containment，并保留 lexical repository receipt path，25/25 tests PASS。

Attempt 02 已消费：72 个冻结 HEAD target 中 71 个返回正 Content-Length；ADAPTER_FIT video `47333152`
的 `lowres_wide.traj` 在 3/3 attempts 均为 HTTP 403，无 Content-Length。响应体读取为 0，可用资产总长度
`1,105,086,109 bytes`。HEAD evidence manifest 已复核；source/work/truth/factor 四个 root 仍不存在，
truth one-shot 未消费。

后续用户授权与 recovery execution 已使锁定 source/truth 路径实际完成；当前 model-free R3 终态为
`TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE`，原因是 complete query truth admission 未满足，
不是 source body 缺失。239 个 DepthART eval candidate 已在 truth join 前封存；描述性 partial-factor
canary 随后完成。新的 Apple scale R0 又在独立两阶段 replay 中先封存 239/239 source-only estimates，
再与 166 帧、1,494 query 的 FARO scale oracle 比较：16/16 parents、163/166 frames 改善，parent-macro
absolute log error 从 `0.3049765` 降至 `0.0156090`。该结果只授权下一步 source-anchored factor
injection canary，不改变正式 O0R NOT_EVALUABLE 终态。

source-anchored R1 随后在全部 171 个 compact-truth frames / 1,539 queries 上完成：成功分支的
SUPPORT height parent-macro error 从 `0.34618 m` 降至 `0.10818 m`，BOUNDARY XYZ error 从
`0.42575 m` 降至 `0.10959 m`；但 hard support extractor 新失去 112 queries / 14 frames，恢复
0 个 baseline failures，且 paired support normal 多数变差。因此 Apple scale estimator 保留，
unconditional pre-extraction injection 不采用。唯一 successor 改为先在这 14 个 lost frames 上做
source-only Apple support seed recovery；无法恢复或一致性不足必须继续 UNKNOWN，不得从同一 eval
cohort 事后选部署阈值。原 R1 summary 的 12-decimal round-trip 末位漂移已由不重算几何的 R1A
reconciliation 封闭；逐帧/query 证据未变。

P0 的 analytic fixture 不是标签清单：validator 会从 measurement-only Jacobian 重算强/弱子空间、
finite task ambiguity 与非光滑分支，并重算 `8 arms × 2 modes × 6 cases = 96` 份
payload/output/common-support hash。通用治理验证的两条 sealed-future-partition warning 已在 result
披露，不构成 O0M/O0R 执行权限。

## 路线隔离与共享边界

- TARO 只能只读复用冻结的相机几何 receipt、R2 factor 字段、deterministic reducer、
  truth-reader 约定、analytic fixture、公开文献和明确标注的数据能力结论。
- 不得写入 Assistive Geometry、AG-QSF、AG-CBF、DepthART/HFTF、RCLE 或 USTRF 的 active
  artifact、checkpoint、progress、optimizer、scheduler、target cache 或 outcome 目录。
- P0 静态 Module 位于 `scripts/research/taro/`；O0M 的独立解析 runtime 位于
  `scripts/research/taro_o0m_runtime/`，不含模型、materializer 或 trainer。唯一 scientific evidence
  固定在 `artifacts.local/evidence/taro/o0m-analytic-mechanics-r0/`，禁止覆盖、删除或重跑。
- O0R 契约 validator 位于 `scripts/research/taro_o0r_source_adapter/`；hash-bound 纯内存实现位于
  `scripts/research/taro_o0r_source_adapter_runtime/`，无 source I/O、model runner 或 artifact writer。
- O0R truth-only preflight 静态 validator 位于 `scripts/research/taro_o0r_truth_preflight/`；它只重算
  exact roster/URL/request digest/binding/budget/authority/root absence，无网络代码或 artifact writer。
- O0R HEAD/source/truth implementation 位于 `scripts/research/taro_o0r_truth_materializer_runtime/`；两个
  production runner 都必须先验证各自未来的 hash-bound execution lock，当前调用只能 fail closed。
- B1 Selection 已消费，Calibration/Confirmation 继续 sealed；A0 anatomy 只能
  `DIAGNOSTIC_ONLY`。TARO 不继承 B1 threshold、seed outcome、best checkpoint 或晋级权限。
- 当前 ARKitScenes TRAIN 原始 K/pose/depth 能力不等于 TARO target 已物化，也不自动获得
  task-query、主动观测或新 Development 权限；任何内容读取必须由后续 source-specific
  protocol 单独授权。

## 与其他新想法的关系

- `TwinScene` 是未来可独立立项的真实锚定 3D 反事实数据/训练引擎；它可以为 TARO 提供
  paired view 与 factor treatment-effect 数据，但不属于 TARO P0，也不能用 synthetic pair
  自动建立真实因果或 Confirmation claim。
- `AC4D` 是未来可独立立项的动态 first-contact 世界信念；它只有在 D44、Kalman/IMM 等
  metric-track oracle 之外证明困难分层增益后，才可能消费 TARO 的 metric posterior。
- TARO 首轮不预测 future pixels、不建 4D occupancy world model、不学习社会反应，也不把
  TwinScene/AC4D 作为组合通过条件。

## 当前允许

- 维护本路线 current 与详细路线指南；
- outcome-blind 的文献去重、接口设计、数据字段映射和解析公式检查；
- 重放 hash-bound O0R adapter 的 44 项、materializer 的 25 项 synthetic tests、6 项 implementation-lock
  mutation tests，以及 truth-only preflight lock 的 8 项静态/mutation tests；
- 只读重放 HEAD receipt/manifest hash、72-row identity、attempt budget 与 zero-body validator；不得发送新请求；
- 只读审计候选公开数据源的文档、许可、字段与能力，并记录 `CANDIDATE_METADATA_MAPPED` /
  `GAP_OPEN` / `NOT_ADMITTED`；该映射不等于下载、source admission 或 O0R successor；
- 重放 P0/O0M 静态 validator、mutation tests 与 O0M runtime unit tests；
- 对已签署 O0M evidence 做只读 hash、manifest、record 与 replay 审计；
- 只读引用历史负结果、数据能力、现有 reducer 和运行时 receipt 的已签署结论。

## 当前禁止

- 从 RGB-only、纯旋转或无独立米制锚的输入输出有限高置信 metric scale；
- 从零学习或覆盖有效的 Camera2/ARCore K、crop、rotation、resize receipt；
- 把不可观测方向、缺字段、track 丢失、动态污染或 unsupported factor 当作 clear/negative；
- 让 learned graph、gauge solver 或 action scorer直接输出最终
  `CLEAR/OCCUPIED/UNKNOWN`，或绕过 deterministic body-swept reducer；
- 把 conformal/CRC 的分布内统计保证写成真实助行安全保证，或用它修复无效 receipt、缺失 metric
  anchor、数据漂移和不满足 exchangeability 的输入；
- 把 VGGT、MapAnything、MASt3R-SLAM 等几何基础模型当作独立米制锚、O0R truth、因果裁判或最终
  三态输出；它们最多是显式 provenance 的 teacher/proposal/initializer/upper bound；
- 用 task metric 选择/拯救 R2 factor backbone checkpoint；
- 要求用户向前或侧向迈步来获取证据，或把计划动作当作已执行基线；
- 读取受保护 outcome、重标 consumed cohort、启动训练/Teacher/TwinScene/AC4D、接
  Android/QNN/HTP、修改默认 App 或宣称助盲安全、独立行走、产品有效性。
- 覆盖、删除或重跑已消费的 O0M one-shot，或把历史 execution lock 解释为剩余权限；
- 把 synthetic mechanics 写成真实 factor causal headroom，或跳过 real O0R 数据/adapter 前门进入
  G0/G1、A0/A1、J0。
- 在独立、已提交且仍受预算/根隔离约束的 truth-only execution authorization 前下载本合同 24 个
  source body、物化 FARO factor/query truth、拟合 uncertainty、运行 DepthART 或生成任一 O0R arm。

## Claim ceiling

当前还证明一个受限算法结果：在锁定 ARKitScenes landscape cohort 和已封存 DepthART candidate 上，
source-visible AppleDepth 零参数尺度锚点能把 parent-macro absolute log-scale error 从 `0.30498` 降到
`0.01561`，239/239 source frames 可估计。它不证明正式 factor causal headroom、RGB-only operation、
真实 evidence dedup/whitening、完整 query truth admission、选择性风险校准、observation-withholding/sensing regret、
真实 geometry-anchored counterfactual pair、被动/主动视角收益、跨设备泛化、移动端可行、产品有效性
或真实用户安全。

默认 App、正式 YOLO、Assistive Geometry 主线、DepthART 路线以及所有产品/安全权限均不变。
