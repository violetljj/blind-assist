# SATOM-A current

状态：`current / WILD_LAB / DEVELOPMENT_STANDARD / SATOM_R0_ACTIVE_REVERSIBLE_EXPLORATION / SYNTHETIC_MECHANICS_ONLY / REAL_E0_NOT_RUN / DEFAULT_APP_UNCHANGED`

## 主张

SATOM-A 检验一个与既有 selector 路线实质不同的问题：新增真实绝对 range observability，
用 metric pose 把 frozen dense prior、主动稀疏 ToF 与历史证据融合为轻量 causal
task-space occupancy memory，并按身体带与 horizon 输出 `CLEAR / OCCUPIED / UNKNOWN`。

它不恢复 TARO、Assistive Geometry、Q-Plane、RCLE、USTRF 或 DepthART D3R6；这些路线
的历史 terminal、暂停状态、数据角色和禁止动作全部不变。

## 当前证据

[SATOM-R0 Module](../../../scripts/research/satom_r0/README.md) 已实现模拟、确定性 polar
evidential memory、五种扫描策略、必要基线、parent 级指标及三项传感器负控。当前只运行
合成 mechanics canary；合成 prior 不是 DepthART 输出，不构成 utility、真实 ToF、设备、
论文创新、产品或安全证据。

## 唯一 successor

`SATOM_R0_REAL_DEPTHART_PRIOR_MULTI_PARENT_E0`：从现有 Bonn RGB-D+pose Development
数据开始，离线物化逐帧 frozen DepthART dense prior 并绑定 provenance，随后在同一
evaluator 中执行最小多-parent E0。不得使用完整 parent 的未来分布选择 ROI、阈值、
预算或 arm。

## 禁止动作

- 不建立 TARO R39/R40/R41，不回调 D3R6 risk score/budget，不恢复 AG/Q-Plane；
- 不把合成/registered RGB-D truth 作为 candidate 输入或 DepthART prior；
- 不在缺少 single-frame、ToF-only、random、round-robin、uniform fusion 时签署正结果；
- deterministic headroom 未成立前不训练 association/refiner/memory/scheduler；
- 不接 Android、不修改默认 App、不建立产品/安全/真实用户或论文 claim。

默认 App 影响：`否`。
