# GA-SATOM current

状态：`current / paused / G0_PROTOCOL_RETAINED / PAUSED_BY_NO_EXTERNAL_TOF_SELECTION / REAL_G0_NOT_RUN / NO_PROCUREMENT / NO_SATOM_ARMS / NO_TRAINING / DEFAULT_APP_UNCHANGED`

用户已进一步选择不引入 IMU、ToF 或已知相机高度的 [SVRF](../svrf/README.md) 作为当前算法主线。
本页协议继续作为未消费备选保留，不采购、不采集、不运行；这不是 G0 失败或路线反证。

## 主张

GA-SATOM 将固定稀疏测距信息预算显式拆成 metric-frame calibration 与未来 task sensing。
G0 只检验第一项：冻结的 `12/64 = 18.75%` VL53L8CX ground-anchor zone pattern，能否在
fresh physical parents 上稳定恢复 RGB camera height，并输出 uncertainty/support/residual。

它不是 SATOM-R0 的 Bonn/DepthART rescue。SATOM-R0 的 consumed inputs、失败高度方法和
0-arm-metric 终态保持关闭；G0 candidate 不加载 DepthART，也不读取机械高度 truth 或
ground labels。

## 当前证据

- [G0 protocol](GA_SATOM_G0_PHYSICAL_GROUND_ANCHOR_PROTOCOL_2026-08-15.json) 已冻结 sensor、
  information budget、anchor pattern、fresh cohort eligibility、独立真值、metrics、winner rule
  和失败即关闭条件；
- [G0 Module](../../../scripts/research/ga_satom_g0/README.md) 已实现 truth-isolated estimator、
  parent-macro/worst-parent evaluator、protocol/activation/hash-bound 双流入口和 mechanics tests；
- 每个 parent 的 9 个 episode 各预注册 300 个时间槽；丢失或不可用采样必须写成全 64-zone
  `INVALID` frame，不能同时从 measurement/truth 删除来抬高 coverage；
- 当前没有 VL53L8CX physical bench、admitted RGB/ToF/IMU registration 或 exact fresh roster，
  所以 outcome access 未激活，`REAL_G0_NOT_RUN`。

外部工作仅支持“极稀疏 metric range 值得检验”的研究前提：CVPR 2026 的
[sparse direct-ToF metric depth completion](https://openaccess.thecvf.com/content/CVPR2026/html/Kim_Dense_Metric_Depth_Completion_from_Sparse_Direct_Time-of-Flight_Sensors_CVPR_2026_paper.html)、
ICCV 2025 的 [ToF-Splatting](https://openaccess.thecvf.com/content/ICCV2025/html/Conti_ToF-Splatting_Dense_SLAM_using_Sparse_Time-of-Flight_Depth_and_Multi-Frame_Integration_ICCV_2025_paper.html)
与 CVPR 2024 的 [variable-sparsity depth completion](https://openaccess.thecvf.com/content/CVPR2024/html/Park_Flexible_Depth_Completion_for_Sparse_and_Varying_Point_Densities_CVPR_2024_paper.html)
均不替代本协议的 G0 物理证据。

## 唯一 successor

无。只有用户明确恢复 external-ToF 路线后，才可重新授权
`GA_SATOM_G0_PHYSICAL_CAPTURE_PREFLIGHT`；现阶段不采购、不采集、不运行 G0。

## 禁止动作

- 不用单区 ToF4M、Bonn simulated ToF、旧 DepthART 输出或 truth height 替代 physical G0；
- 不把 VL53L8CX 全帧采集的 64 zones 写成 12-zone 物理功耗预算；当前只冻结 algorithmic
  information budget；
- 不在 G0 运行 SATOM arm、比较 scheduler、训练 selector、进入 G1/Android 或修改默认 App；
- 不在打开 fresh range outcome 后更换 parent、anchor pattern、门或 reference truth。

默认 App 影响：`否`。
