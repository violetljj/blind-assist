# DA V2 模型变体准确率与 false-clear 门 R1

日期：2026-08-05（仅对尚未产生输出的未来候选前瞻生效）

## 结论

R1 保留 R0 的固定 120 帧 roster、深度误差、地面恢复、clearance、collision、false-clear
和时序门，但用传感器真值替换三项 canonical 相对硬门：VALID/UNKNOWN、完整几何状态、
连续帧变化。它同时新增 false-block 与相对基线的 harmful/beneficial decision change，防止
轻量模型靠“全占用”把 false-clear 做低。

R0 的 A1、A2、A3 结论已经消费，保持原终点，禁止用 R1 重算或翻案。R1 的目的只是使
后续候选的门更准确，而不是救回已失败模型。

## 为什么需要 R1

canonical DA V2 在该 cohort 的 false-clear 为 `203/837 = 24.25%`。因此，候选若纠正
canonical 的错误，其几何状态本来就应当不同；R0 的 canonical state/transition exact
agreement 会把这种改善也当成回归。另一方面，A3 证明只盯 false-clear 会容许“几乎全占用”
的保守塌缩。

R1 将安全相关判断直接对齐传感器真值：

- truth status exact agreement：较 canonical 下降不超过 `1 pp`；
- truth 3 band × 3 horizon state exact agreement：下降不超过 `2 pp`；
- truth 连续帧变化/不变 agreement：下降不超过 `5 pp`；
- false-block 占全部已知决策的比例：较 canonical 增加不超过 `2 pp`；
- 从 baseline 正确变为 candidate 错误的 harmful change：不超过全部已知决策 `2%`；
- beneficial changes 数量必须不少于 harmful changes。

以上与继承的八项 R0 非劣化门全部 AND。canonical 相对状态差异仍可报告，但不再作为
未来候选的硬门。

## 证据边界与执行顺序

该 TUM cohort 已被 Development 工作消费，只能提供工程回归和 Pareto 证据，不能建立
产品或安全 authority。未来候选必须先冻结协议、模型、训练与缓存物化规则，再一次性生成
完整深度缓存并运行 R1；只有通过质量门，才允许做真机端到端 P95 profile。最终相机门仍需
新的 session/parent-disjoint RGB-D 或量距真值。

机器可读权威为同名 JSON；执行器为
`scripts/research/hftf/evaluate_dav2_model_variant_gate_r1.py`，其冻结 SHA-256 为
`BACA2514F557CE8104994C50353881B359BEF62880BAFFF4FC223875AFFE20E2`。
