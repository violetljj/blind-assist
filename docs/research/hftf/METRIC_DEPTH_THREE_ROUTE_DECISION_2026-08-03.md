# Metric depth three-route decision

日期：2026-08-03

决策：`ALL_THREE_CURRENT_CANDIDATES_NOT_SUPPORTED_ON_CONSUMED_DEVELOPMENT__TOF_PURCHASE_HOLD`

## 结论

三条路线都已按预先冻结的窄实现落地并回放，但当前没有一条获得替代 ToF 或进入手机
实时方案的证据：

| 路线 | 关键结果 | 冻结终态 |
| --- | --- | --- |
| 1. 异步关键帧 affine | 58/120 已知，MAE 0.15241 m，一致率 86.21%，false-clear 8.43% | task 与 shared-HTP feasibility 均不支持 |
| 2. 稠密 residual + RAFT | 81/120 paired，MAE 0.17694 m，一致率 89.81%，false-clear 7.02% | 不支持 |
| 3. 离线教师 770 参数 head | MAE 0.19980 m，4/4 胜 raw、2/4 胜常数，一致率 89.67%，false-clear 8.92% | 不支持 |

路线二把已知覆盖从路线一的 48.33% 提高到 67.5%，并保持零因果违规，但仍没有跨过
任务门，且双向 RAFT 把 PC 稳态输出 P95 推到 `131.59 ms`。路线三最接近“手机只跑
DA+小头”的目标，也显著优于 raw DA；但当前单 CLS/global-affine 表示没有稳定胜过
简单常数校准，更不能用平均 MAE 掩盖 false-clear 失败。

因此当前行动规则是：

1. 停止在已消费 TUM 上调整这三条候选；保留 Metric3D 作为 PC/离线教师。
2. ToF 采购继续 `HOLD`，因为这些是窄实现的 consumed Development 负结果，不是
   “纯 RGB 原理不可能”的证明。
3. 若继续纯 RGB，优先级高于再做在线双模型的是：在新 protocol 下收集/指定
   parent/session-disjoint RGB-D 或测距 cohort，冻结一个具备空间/分区信息但仍满足端侧
   参数预算的学生头，再做 fresh 评价。
4. 若 fresh 学生仍在 false-clear、跨域或视觉共模误差门失败，则预冻结 VL53L5CX E 臂，
   与最佳纯 RGB 候选在同一最终摄像头、同一 session ledger 下比较。

## 证据强弱

强证据是实现级因果约束、hash-bound 输入/cache、无 outcome 后搜索、四序列 LOSO 和
明确负终点。弱证据是数据量小且已消费、sensor 只是几何 proxy、没有最终摄像头或手机
共驻测量、没有透明/镜面/强光等独立失效族群。故既不能宣称 ToF 必需，也不能宣称纯
RGB 已足够。

对应结果：

- [方案一](DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_DEVELOPMENT_RESULT_2026-08-03.md)
- [方案二](DENSE_METRIC_DEPTH_PROPAGATION_R0_DEVELOPMENT_RESULT_2026-08-03.md)
- [方案三](METRIC_DEPTH_CALIBRATION_HEAD_DISTILLATION_R0_DEVELOPMENT_RESULT_2026-08-03.md)

