# Assistive Geometry D0 时序消融协议

状态：`MECHANICS_FROZEN_SINGLE_FRAME_BASELINE_AND_NEW_COHORT_NOT_AVAILABLE`

本协议在任何时序数据 outcome 前固定同一个八帧 GeometryState 输入、48 维隐藏状态和
`50k` 参数上限，比较 GRU、因果 TCN 与明确不作 Mamba 主张的稳定 diagonal SSM。三者只输出
未来 `0.25/0.5/1.0 s` clearance delta、band TTC 和 raw compute-gate logit；最终三态、UNKNOWN
原因与跳帧安全约束继续由 host postprocess 掌握。测试会用未来帧扰动验证前缀输出不变，且
显式 neutralize `state_known=false` 的 occupancy payload，禁止把 UNKNOWN 编码成 clear。

激活前仍缺稳定的 B2/C1 单帧候选、新的至少 8-parent `TEMPORAL_DEVELOPMENT` cohort、密封的
parent/session-disjoint Confirmation roster、future-clearance/TTC truth materializer 以及训练/评价
runner。三 seed 不选 best seed；只有覆盖率、false-clear/false-block、temporal delta、future
clearance、TTC 事件支持和真实设备 `5 ms P95` 预算全部通过的 arm 才可进入选择。若无合格 arm，
终态固定为 `ASSISTIVE_GEOMETRY_D_TEMPORAL_NOT_SUPPORTED_RETAIN_SINGLE_FRAME`。

当前不授权打开时序数据、训练、Development、Confirmation、部署、默认 App、产品或 safety。
