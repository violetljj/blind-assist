# DA V2 模型变体门 R2 canonical 自检

日期：2026-08-05

canonical 缓存与自身比较仍通过全部 `14/14` 门，且未定义指标为 0，证明 R2 对正常有限
输入保持 R1 判定。R2 的唯一新增行为是：门控指标为 `null`、NaN 或 infinity 时直接失败。

完整结果位于忽略目录
`artifacts.local/evidence/hftf/dav2-model-optimization-p2-r0/p1-r2-canonical-self-check.json`，
SHA-256 为 `AC84FE98...E0AE`。这只是 evaluator self-check，不改变 baseline 的准确率、
false-clear 或绝对任务终态。
