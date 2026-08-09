# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / P0_PASS / O0M_PROTOCOL_FROZEN / O0M_IMPLEMENTATION_LOCK_PASS / SCIENTIFIC_STATUS_NOT_RUN / O0M_EXECUTION_NOT_AUTHORIZED / O0R_NOT_EVALUABLE_DATA_AND_INTERFACE / DEFAULT_APP_UNCHANGED`

## 当前主张

TARO（Task-directed Active Risk Observability，任务定向主动风险可观测性）研究：

> 在有效相机/裁剪/旋转 receipt 与至少一个独立米制锚存在时，即使完整场景与完整
> gauge 不能被唯一恢复，body-swept clearance 查询能否先达到可识别、可校准状态；
> 当查询仍不可识别时，受限的被动帧复用或站定相机微基线，能否比通用熵、最大视差
> 或普通 next-best-view 更有效地降低任务决策风险，并在证据不足时保持 `UNKNOWN`？

本路线把两个已讨论组件合并为一个不可拆分的论文命题：

- `GaugeFix`：metadata-first 的低维残余 gauge posterior、协方差与可观测子空间更新；
- `PARA`：以 body/path-specific clearance query 为目标的受限主动视差与证据选择。

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
  与 13/13 disjoint unit tests 已 hash-bound；正式 execution family 仍未运行；
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)：项目级算法路线登记；
- [R2 factorized geometry protocol](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)：
  可只读复用的 factor/reducer/UNKNOWN 上游合同；
- [A0 failure anatomy](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)：
  只作为系统性 conservative-bias 诊断，不作为 TARO 选模或阈值来源；
- [AG-DCA result](../assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.md)：
  当前 target-atlas timestamp/pose/factor-support 缺口的能力真源。

## 唯一 successor

`TARO_O0M_ONE_SHOT_EXECUTION_LOCK`

O0M implementation 已锁定且 13/13 disjoint tests 通过。该 successor 只允许另提交 one-shot
execution lock，把 fixture/code/tests/command/environment/timeout/exclusive absent root 全部绑定；锁提交
前不得创建 scientific artifact 或运行正式 10+80+2 canary。

真实 O0R 当前硬终态为 `TARO_O0R_NOT_EVALUABLE_DATA_AND_INTERFACE`：complete factor truth、
truth-clear factor bundle、连续 boundary/uncertainty truth、target timestamp/pose、deterministic
injection adapter 和 fresh paired outcome 均未满足。Synthetic O0M PASS 也不能改写该终态。

P0 的 analytic fixture 不是标签清单：validator 会从 measurement-only Jacobian 重算强/弱子空间、
finite task ambiguity 与非光滑分支，并重算 `8 arms × 2 modes × 6 cases = 96` 份
payload/output/common-support hash。通用治理验证的两条 sealed-future-partition warning 已在 result
披露，不构成 O0M/O0R 执行权限。

## 路线隔离与共享边界

- TARO 只能只读复用冻结的相机几何 receipt、R2 factor 字段、deterministic reducer、
  truth-reader 约定、analytic fixture、公开文献和明确标注的数据能力结论。
- 不得写入 Assistive Geometry、AG-QSF、AG-CBF、DepthART/HFTF、RCLE 或 USTRF 的 active
  artifact、checkpoint、progress、optimizer、scheduler、target cache 或 outcome 目录。
- P0 已创建只做静态合同检查的 `scripts/research/taro/` Module；其中没有 solver、runner、
  materializer、模型或 trainer。未来执行仍须独占
  `artifacts.local/{work,models,evidence}/taro/`，当前没有 TARO scientific artifact。
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
- 重放 P0 静态 validator 与 mutation tests；
- 起草唯一 O0M successor 的非执行 protocol/implementation lock；
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
- 在 O0M execution lock 前创建/运行 solver、factorial oracle runner 或 scientific artifact；
- 把 synthetic mechanics 写成真实 factor causal headroom，或跳过 real O0R 数据/adapter 前门进入
  G1、A0/A1、J0。

## Claim ceiling

当前证明 P0 machine contracts、解析期望、权限和 route failure boundary 静态自洽；没有执行
task-query identifiability、factor causal headroom、residual gauge、主动微基线或真实数据实验，
也没有证明学生可学、跨设备泛化、移动端可行或真实用户安全。

默认 App、正式 YOLO、Assistive Geometry 主线、DepthART 路线以及所有产品/安全权限均不变。
