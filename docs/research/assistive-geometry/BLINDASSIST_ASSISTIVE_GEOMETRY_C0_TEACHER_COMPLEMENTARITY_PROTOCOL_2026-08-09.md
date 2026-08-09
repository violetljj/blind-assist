# Assistive Geometry C0 异质教师互补性协议

状态：`MECHANICS_FROZEN / COHORT_AND_TEACHER_OUTPUTS_NOT_AUTHORIZED`

C0 不直接训练双教师 student。它先在新的 parent/session-disjoint `TEACHER_EVALUATION`
cohort 上回答两个 teacher 是否真的互补。teacher/pseudo-label 永远不是独立 truth，`UNKNOWN`
也永远不是 negative；teacher 输出必须先于 truth join 生成。

kill gate 同时要求：oracle 相对最佳单教师至少改善 clearance 5% 或 false-clear 0.01；两个
teacher 都在至少两个独立 parent 上有 exclusive-correct；disagreement 区域的真实 error rate
比 agreement 高至少 0.10；temporal/geometry teacher 的 temporal clearance delta MAE 至少改善
0.01m。任一失败即关闭对应 C1 异质蒸馏主张。

当前只冻结 evaluator mechanics。teacher 精确身份、checkpoint、license、变换和新 cohort 都是
`UNRESOLVED`，因此 teacher 执行和 C1 训练仍未授权。
