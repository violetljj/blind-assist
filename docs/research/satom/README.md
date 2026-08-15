# SATOM-A current

状态：`current / WILD_LAB / DEVELOPMENT_STANDARD / SATOM_R0_REAL_E0_NOT_EVALUABLE / DEPTHART_GROUND_HEIGHT_OBSERVABILITY_FAIL / NO_ARM_METRIC / CLOSED_NO_TUNING / DEFAULT_APP_UNCHANGED`

## 主张

SATOM-A 检验一个与既有 selector 路线实质不同的问题：新增真实绝对 range observability，
用 metric pose 把 frozen dense prior、主动稀疏 ToF 与历史证据融合为轻量 causal
task-space occupancy memory，并按身体带与 horizon 输出 `CLEAR / OCCUPIED / UNKNOWN`。

它不恢复 TARO、Assistive Geometry、Q-Plane、RCLE、USTRF 或 DepthART D3R6；这些路线
的历史 terminal、暂停状态、数据角色和禁止动作全部不变。

## 当前证据

[SATOM-R0 Module](../../../scripts/research/satom_r0/README.md) 已实现模拟、确定性 polar
evidential memory、五种扫描策略、必要基线、parent 级指标、matched-coverage/Pareto 诊断及
三项传感器负控。Bonn roster、DepthART prior、PRIMARY 和 winner rule 已在 outcome access
前冻结，但 [Real E0](SATOM_R0_BONN_REAL_E0_NOT_EVALUABLE_2026-08-15.md) 在 arm metric 前
因 DepthART ground-height observability 不稳定而 `NOT_EVALUABLE`。合成 prior 仍不是
DepthART utility 证据；Real E0 没有算法结果。

## 唯一 successor

无。状态为 `NONE / SATOM_R0_CLOSED_AFTER_REAL_E0_NOT_EVALUABLE`。只有新的 pre-outcome
协议先提供 source-native ground height、独立 metric height observability，或不依赖绝对
ground height 的 materially different task representation 后，才可重开；不得在已打开的
Bonn/DepthART 输出上继续修改高度算法或 winner rule。

## 禁止动作

- 不建立 TARO R39/R40/R41，不回调 D3R6 risk score/budget，不恢复 AG/Q-Plane；
- 不把合成/registered RGB-D truth 作为 candidate 输入或 DepthART prior；
- 不在缺少 single-frame、ToF-only、random、round-robin、uniform fusion 时签署正结果；
- deterministic headroom 未成立前不训练 association/refiner/memory/scheduler；
- 不接 Android、不修改默认 App、不建立产品/安全/真实用户或论文 claim。

默认 App 影响：`否`。
