# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / R0_ROUTE_GUIDE_FROZEN / DESIGN_AUTHORIZED / EXECUTION_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

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
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)：项目级算法路线登记；
- [R2 factorized geometry protocol](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)：
  可只读复用的 factor/reducer/UNKNOWN 上游合同；
- [A0 failure anatomy](../assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)：
  只作为系统性 conservative-bias 诊断，不作为 TARO 选模或阈值来源；
- [AG-DCA result](../assistive-geometry-data-capability/BLINDASSIST_ASSISTIVE_GEOMETRY_DCA_R0_RESULT_2026-08-10.md)：
  当前 target-atlas timestamp/pose/factor-support 缺口的能力真源。

## 唯一 successor

`TARO_P0_TASK_QUERY_IDENTIFIABILITY_AND_FACTOR_ORACLE_CANARY_PROTOCOL_LOCK`

该 successor 只允许另行冻结一个 `CANARY_LITE`、outcome-blind、非执行协议，内容限于：

1. `TaroFrameReceipt`、`TaroFactorPosterior`、`TaroTaskQuery` 和候选观测动作 schema；
2. task-query local identifiability、Fisher/Hessian null-space 与 TSVD 更新规则；
3. 解析 synthetic fixture、退化运动负控与 factor-oracle factorial arms；
4. 数据角色、输入 provenance、`max_source_timestamp`、UNKNOWN reason 与 artifact namespace；
5. primary metric、拟议 gate 的依据、failure scope、资源预算与停止条件。

P0 本身不授权实现、canary 执行、真实数据读取、factor 注入、训练、模型、checkpoint、
Development outcome、主动用户提示、设备测量或 Android 集成。P0 protocol 只有在单独完成并
把 execution authority 显式改为 true 后，才可能产生一次 O0 mechanics/oracle canary；
本页不预授权 O0 或任何后续阶段。

## 路线隔离与共享边界

- TARO 只能只读复用冻结的相机几何 receipt、R2 factor 字段、deterministic reducer、
  truth-reader 约定、analytic fixture、公开文献和明确标注的数据能力结论。
- 不得写入 Assistive Geometry、AG-QSF、AG-CBF、DepthART/HFTF、RCLE 或 USTRF 的 active
  artifact、checkpoint、progress、optimizer、scheduler、target cache 或 outcome 目录。
- 若未来产生实现，计划独占 `scripts/research/taro/` 与
  `artifacts.local/{work,models,evidence}/taro/`；当前尚未创建代码 Module 或 artifact。
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
- 起草唯一 P0 successor 的非执行协议；
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

## Claim ceiling

当前只证明用户已选择 TARO 作为独立并行研究支线，并冻结了可证伪路线、依赖顺序、
权限和停止边界。没有证明 task-query identifiability、factor causal headroom、residual gauge
可估计、主动微基线有价值、学生可学、跨设备泛化、移动端可行或真实用户安全。

默认 App、正式 YOLO、Assistive Geometry 主线、DepthART 路线以及所有产品/安全权限均不变。
