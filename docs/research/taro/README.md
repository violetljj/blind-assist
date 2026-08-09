# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / P0_PASS / O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS / O0M_ONE_SHOT_CONSUMED / O0R_NOT_EVALUABLE_DATA_AND_INTERFACE / PAUSED_NO_ACTIVE_EXECUTION / DEFAULT_APP_UNCHANGED`

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
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)：项目级算法路线登记；
- [R2 factorized geometry protocol](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)：
  可只读复用的 factor/reducer/UNKNOWN 上游合同；
- [A0 failure anatomy](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)：
  只作为系统性 conservative-bias 诊断，不作为 TARO 选模或阈值来源；
- [AG-DCA result](../assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.md)：
  当前 target-atlas timestamp/pose/factor-support 缺口的能力真源。

## 唯一 successor

无：`PAUSED_NO_ACTIVE_EXECUTION`

O0M implementation、one-shot lock 与唯一正式执行均已完成。正式 `10+80+2` synthetic canary
终态为 `TARO_O0M_SYNTHETIC_ANALYTIC_MECHANICS_PASS`；exclusive root 已创建并消费，结果不得
覆盖、删除或重跑。

真实 O0R 当前硬终态为 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`：complete factor truth、
truth-clear factor bundle、连续 boundary/uncertainty truth、target timestamp/pose、deterministic
injection adapter 和 fresh paired outcome 均未满足。Synthetic O0M PASS 不能改写该终态，也不授权
`G0/G1/A0/A1/J0`。只有新的 pre-outcome source-and-adapter contract 同时满足全部 O0R 前门后，
才可另立冻结路线版本；当前没有隐含 successor。

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
- 重放 P0/O0M 静态 validator、mutation tests 与 O0M runtime unit tests；
- 对已签署 O0M evidence 做只读 hash、manifest、record 与 replay 审计；
- 只读引用历史负结果、数据能力、现有 reducer 和运行时 receipt 的已签署结论。

## 当前禁止

- 从 RGB-only、纯旋转或无独立米制锚的输入输出有限高置信 metric scale；
- 从零学习或覆盖有效的 Camera2/ARCore K、crop、rotation、resize receipt；
- 把不可观测方向、缺字段、track 丢失、动态污染或 unsupported factor 当作 clear/negative；
- 让 learned graph、gauge solver 或 action scorer直接输出最终
  `CLEAR/OCCUPIED/UNKNOWN`，或绕过 deterministic body-swept reducer；
- 用 task metric 选择/拯救 R2 factor backbone checkpoint；
- 要求用户向前或侧向迈步来获取证据，或把计划动作当作已执行基线；
- 读取受保护 outcome、重标 consumed cohort、启动训练/Teacher/TwinScene/AC4D、接
  Android/QNN/HTP、修改默认 App 或宣称助盲安全、独立行走、产品有效性。
- 覆盖、删除或重跑已消费的 O0M one-shot，或把历史 execution lock 解释为剩余权限；
- 把 synthetic mechanics 写成真实 factor causal headroom，或跳过 real O0R 数据/adapter 前门进入
  G0/G1、A0/A1、J0。

## Claim ceiling

当前只证明冻结的、预去重与预白化 synthetic analytic family 上，独立 NumPy 实现可复现
task-query identifiability 与 factorial intervention mechanics。它不证明真实 factor causal headroom、
真实 evidence dedup/whitening、模型质量、被动/主动视角收益、跨设备泛化、移动端可行、产品有效性
或真实用户安全。

默认 App、正式 YOLO、Assistive Geometry 主线、DepthART 路线以及所有产品/安全权限均不变。
