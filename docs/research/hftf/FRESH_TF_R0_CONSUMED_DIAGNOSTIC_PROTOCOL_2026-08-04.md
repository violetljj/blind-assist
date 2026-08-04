# FRESH-TF R0 consumed diagnostic protocol

日期：2026-08-04
证据角色：`CONSUMED_DIAGNOSTIC_ONLY`
执行前状态：`FROZEN_BEFORE_ARM_METRICS`

## 决策与继承边界

FRESH-TF 记录为 HFTF 的受控后继候选。它复用已经建立的 `foot/body/head` 人体扫掠
包络、米制场、三态 `UNKNOWN` 和 provenance；这些 inherited primitives 不重新申报
为本路线的新颖性。

本路线待证的最小差异是：在低频精确米制场与高频当前帧之间，能否显式估计每个决策
单元的有效期，使选择性 `CLEAR` 相对零阶保持降低 stale false-clear，并在固定深度
调用预算下保留有用 coverage。风险触发 NPU 调度只有在该表示增量成立后才能进入 R1。

## 本次首试

输入固定为已消费的 Bonn person-tracking 30 帧同步 RGB-D teacher sidecar。它在本协议
之前已经生成并可见，因此只能做单 parent-sequence 机制诊断，不能成为 fresh、独立复现
或论文效果证据。

精确场按因果 0.5 秒 cadence 取 anchor。每个当前帧的 RGB 只计算固定的
`64x48 grayscale MAD / 255` 场景变化量；当前 registered depth 只作评价真值，不能进入
候选臂。固定四臂：

1. 2 Hz zero-order hold；
2. 2 Hz + 750 ms hard TTL；
3. 统一 age freshness，`tau=0.5 s`；
4. selective RGB-change freshness，`tau_clear=0.35 s`、`tau_blocked=0.75 s`。

统一 quality threshold 为 `0.5`，RGB-change scale 为 `0.08`。低于门限只能
`UNKNOWN`，不得转成 `CLEAR`。精确数值、方向、距离 horizon、continuation gates 和输入
hash 见同名 JSON；结果后不得搜索或修改这些值来救援。

## 继续门

只有以下条件全部满足，终态才允许为 consumed diagnostic mechanism signal：

- anchor 到当前真值至少产生 10 个 cell-state transitions；
- selective false-clear count 严格低于 zero-order hold；
- selective false-clear rate 不劣于 uniform freshness；
- selective known coverage 至少 65%；
- selective known accuracy 相对 TTL 的非劣界为 2 个百分点；
- `UNKNOWN -> CLEAR` violation 为 0。

机会不足时为 `NOT_EVALUABLE`；存在足够机会但任一继续门失败时为
`NOT_SUPPORTED`。无论结果如何，本次均不评价 height-stratified 增量、warp、NPU 主动
调度、能耗、App 提醒、导航或安全效果。
