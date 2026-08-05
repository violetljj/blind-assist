# DA V2 模型变体门 R2：未定义指标 fail-closed

日期：2026-08-05

状态：`FROZEN_AFTER_R1_UNDEFINED_METRIC_PREFLIGHT_BEFORE_R2_OUTPUT`

A4-BS25 的唯一缓存进入 R1 后，候选 `clearance_mae_m` 不可定义；R1 在执行
`None <= finite threshold` 时抛出 `TypeError`，没有写出结果。该异常不是质量通过或失败
终态，也不授权更改候选。

R2 只做一个机械修复：任何参与门控的未定义或非有限指标都确定性判失败，并在 JSON 中
写为 `null`。其余有限指标逻辑、14 个门、阈值、真值构造、roster、baseline 和历史终态
全部继承 R1，不作修改。

执行前必须先以 canonical baseline 做 self-check，确认仍为 14/14；之后才允许对已经锁定
SHA-256 `6854CDA8...FD81` 的 A4-BS25 缓存生成 R2 终态。R2 不得用于重标已有结果，
也不得把未知/无决定解释为 false-clear 为零或安全证据。
