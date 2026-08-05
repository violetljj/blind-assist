# DA V2 模型变体门 R1 canonical 自检

日期：2026-08-05

## 结论

canonical 深度缓存与自身比较，通过 R1 全部 `14/14` 工程门，证明协议、哈希绑定、真值几何
重算和门逻辑可以闭环。这个 PASS 只是基线自洽性测试，不改变 canonical 未通过绝对 task 门
的事实，也不授权重标 A1、A2 或 A3。

## 基线真值画像

- truth VALID/UNKNOWN exact：`99.17%`；
- truth 完整几何状态 exact：`2.52%`（`3/119`）；
- truth 连续帧变化/不变 agreement：`55.65%`；
- false-clear：`203/837 = 24.25%`；
- false-block：`4/837 = 0.48%`；
- canonical 自比 harmful / beneficial changes：`0 / 0`。

低完整状态 exact 与高 false-clear 共同证明：后续候选不能继续把 canonical 状态一致性作为硬门；
低 false-block 又说明，A3 那类全占用塌缩必须被单独拦截。

完整执行结果保存在忽略目录
`artifacts.local/evidence/hftf/dav2-model-variant-gate-r1/canonical-self-check.json`，SHA-256 为
`BFA6CE06D37ED41D020326F5FEA89938C6FCC91101096853247912FA318E9309`。可提交的机器可读摘要为
同名 JSON receipt。
