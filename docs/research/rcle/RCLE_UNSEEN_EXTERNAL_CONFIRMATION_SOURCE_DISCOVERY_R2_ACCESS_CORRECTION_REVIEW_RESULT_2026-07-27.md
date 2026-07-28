# RCLE unseen external confirmation Source Discovery R2 access-correction review

日期：2026-07-27

## 结论

`SOURCE_DISCOVERY_R2_ACCESS_CORRECTION_REVIEW_PASS / EXECUTION_AUTHORIZED`

本 review 确认 R2 只纠正 agent 在 R1 中加入的机械访问限制，不改变用户冻结的候选、
窗口、算法或科学门槛。

- access correction：
  [RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R2_ACCESS_CORRECTION_2026-07-27.json](RCLE_UNSEEN_EXTERNAL_CONFIRMATION_SOURCE_DISCOVERY_R2_ACCESS_CORRECTION_2026-07-27.json)
- correction SHA-256：
  `849f211a4853653d93e27d1be18315aa370cb5d0e570cf1a1014b1062672730a`
- risk-based standard：
  [RCLE_EVIDENCE_ACCESS_AND_TRANSPORT_STANDARD_R1_2026-07-27.md](RCLE_EVIDENCE_ACCESS_AND_TRANSPORT_STANDARD_R1_2026-07-27.md)
- standard SHA-256：
  `2bb2537ec14a5dfc7623cd3e8161e6f57e799d88824345fc278b6357d0e1e53f`
- review receipt SHA-256：
  `bac2ee7da1742fdca1a36b53227e70b4fe9b44119ba5941c3e643d19c1b25a61`
- errors：`[]`

## 已优化的规则

- compressed object/range/block 到达不再等于 RGB 内容使用；
- solid decoder 可经过技术上不可分离的 color members，但只能在内存中立即丢弃；
- 完整 immutable archive 在许可、identity 和原 `40 GiB` 总预算内可作为 transport；
- range、member、solid block、完整 archive、resume/cache 和 decoder 切换属于等价
  工程实现，不再为每次切换另立科学协议；
- 可恢复的 transport 事件不再立即关闭整个 candidate order；
- 两个已锁定来源独立完成，只有合理且授权的路径耗尽才可报 transport
  `NOT_EVALUABLE`。

## 保持不变

独立 stdlib review 重新绑定 candidate lock 与 geometry producer SHA，并确认以下内容
全部未变：

- 来源仅为 OpenLORIS corridor 与 MultiScan，exact captures、顺序和 identity 不变；
- 固定非重叠 `10 s` 网格、同 capture `20 s` 间隔和 tie-break 不变；
- `>=90` pairs、`dt<=0.1 s`、coverage `>=0.8` 与 positive/below 各项门槛不变；
- fixed denominator、每来源 `1 positive + 1 below`、禁止 pooled rescue 不变；
- geometry formula SHA
  `0ce6256e12dd4536f284c7047f1e63faf955fa7bcf87f28fcb93c3e5d9de1add`
  不变；
- 不扩候选、不降门、不滑窗、不查看或运行 RGB、不启动 Android。

## 当前授权

R1 的历史终态和 receipt 保留，但不再作为当前执行 authority。R2 已允许在上述边界内
恢复 OpenLORIS solid geometry extraction 与 MultiScan depth acquisition，并运行
geometry-only 固定选窗。
